import torch
import torch.nn as nn

def compute_lipschitz_proxy(module, x, eps=1e-3):
    """
    Estimates the local Lipschitz constant: ||f(x+δ) - f(x)|| / ||δ||
    """
    # Ensure module is in eval mode for consistent behavior during check
    training_state = module.training
    module.eval()
    
    with torch.no_grad():
        # 1. Create random perturbation delta
        delta = torch.randn_like(x) * eps
        
        # 2. Compute f(x) and f(x+delta)
        out_original = module(x)
        out_perturbed = module(x + delta)
        
        # 3. Compute norms
        diff_norm = torch.norm(out_perturbed - out_original)
        delta_norm = torch.norm(delta)
        
        # Avoid division by zero
        lipschitz = diff_norm / (delta_norm + 1e-9)
    
    # Restore original training state
    module.train(training_state)
    return lipschitz.item()

def compute_jacobian_sv(module, x, iters=5, eps=1e-3):
    """
    Estimates top singular value of Jacobian J = dy/dx using Power Iteration.
    Jv (Forward) via Finite Difference.
    vJ (Backward) via Autograd.
    """
    training_state = module.training
    module.eval()
    
    # Initialize random vector v (same shape as input)
    v = torch.randn_like(x)
    v = v / (torch.norm(v) + 1e-9)
    
    sigma = 0.0
    
    for _ in range(iters):
        # --- Step 1: Compute u = J * v (Forward approximation) ---
        # Jv ≈ (f(x + eps*v) - f(x)) / eps
        with torch.no_grad():
            x_perturbed = x + eps * v
            out_original = module(x)
            out_perturbed = module(x_perturbed)
            u = (out_perturbed - out_original) / eps
            
        # --- Step 2: Compute v_new = J^T * u (Backward exact) ---
        # Use Vector-Jacobian Product (VJP) via autograd
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = module(x_in)
            
            # grad(outputs, inputs, grad_outputs) computes J^T * grad_outputs
            v_new = torch.autograd.grad(
                outputs=out, 
                inputs=x_in, 
                grad_outputs=u, 
                retain_graph=False,
                create_graph=False
            )[0]
            
        # --- Step 3: Normalize ---
        v_new_norm = torch.norm(v_new)
        if v_new_norm > 1e-9:
            v = v_new / v_new_norm
        
        # Sigma estimate is the norm of u (Rayleigh quotient approx)
        sigma = torch.norm(u).item()

    module.train(training_state)
    return sigma

def run_stability_diagnostics(model, inputs):
    """
    Runs diagnostics on all 'DynamicTanh' modules in the model using 'inputs'.
    """
    results = {}
    layer_inputs = {}
    handles = []
    
    # 1. Hook to capture inputs for each DyT layer
    def get_input(name):
        def hook(model, input, output):
            # Input is a tuple, we take the first tensor
            layer_inputs[name] = input[0].detach() 
        return hook

    # 2. Register hooks
    for name, m in model.named_modules():
        if "DynamicTanh" in str(type(m)):
            handles.append(m.register_forward_hook(get_input(name)))
            
    # 3. Forward pass to populate layer_inputs
    # (Ensure we are in eval mode so we don't update BN statistics or dropout)
    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(inputs)
    model.train(training_state)
    
    # 4. Remove hooks
    for h in handles: h.remove()
    
    # 5. Run tests on captured inputs
    for name, m in model.named_modules():
        if name in layer_inputs:
            inp = layer_inputs[name]
            
            # Lipschitz
            lip = compute_lipschitz_proxy(m, inp)
            
            # Jacobian Spectral Norm
            jac_sv = compute_jacobian_sv(m, inp, iters=5)
            
            results[f"{name}.lipschitz"] = lip
            results[f"{name}.jacobian_sv"] = jac_sv
            
    return results