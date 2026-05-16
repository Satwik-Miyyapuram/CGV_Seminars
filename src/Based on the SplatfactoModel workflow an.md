Based on the SplatfactoModel workflow and the specific
architecture of "Gaussian Meets Surfel" (GES), you should tackle 
the implementation in a phased approach. Because GES introduces  
a dual-representation (Surfels + Gaussians) and a two-pass       
rendering pipeline, building it sequentially and testing as you  
go will save you a lot of debugging time.

Here is the recommended order in which you should write the code 
in src/ges_model.py:

Phase 1: Data Structures & Initialization
Focus on getting the data into the model and exposed to the      
optimizer.

  1. populate_modules():
    * Load Seed Points: Write the logic to initialize the        
      positions (means) from the SFM point cloud (usually found  
      in self.seed_points).
    * Initialize Surfels (2D): Initialize the Surfel tensors     
      (surfel_means, surfel_scales [only 2D!], surfel_quats,     
      surfel_opacities, surfel_features).
    * Initialize Gaussians (3D): Initialize the Gaussian
      tensors. Note: You need to decide if they start empty and  
      grow later, or if they are initialized alongside the       
      surfels.
    * Setup Strategy: Initialize the densification strategy.     
      Because you have two sets of parameters, you cannot easily 
      use a single gsplat.DefaultStrategy. You should set up two 
      separate strategy objects (one for surfels, one for        
      Gaussians) or write a custom manager.

  2. get_param_groups() & get_gaussian_param_groups():
    * Ensure all your torch.nn.Parameter tensors are correctly   
      mapped to strings so the Nerfstudio optimizers can find    
      and update them.
Phase 2: The Forward Pass (Two-Pass Rendering)
Focus on the core GES contribution: rendering the dual
representation.

  3. get_outputs():
    * Camera Setup: Call get_viewmat() to get the camera
      matrices.
    * Pass 1 (Surfels): Call gsplat.rasterization() using only   
      the Surfel parameters. Extract the resulting RGB and       
      Depth.
    * Pass 2 (Gaussians): Call gsplat.rasterization() using the  
      Gaussian parameters. Crucial step: You need to pass the    
      depth from Pass 1 into this pass so the Gaussians are      
      occluded/blended correctly according to the GES paper's    
      $\tau$ modulation. (Note: Check gsplat documentation to    
      see if it supports passing a depth buffer for occlusion,   
      or if you must implement this blending manually in PyTorch 
      after rasterizing the Gaussians).
    * Compositing: Combine the Surfel and Gaussian outputs       
      according to Equation 5 in the paper.
    * Return: Return a dictionary with "rgb", "depth", etc.      

Phase 3: Losses & Metrics
Focus on getting the model to actually learn from the rendered   
images.

  4. get_loss_dict():
    * Implement the standard L1 Loss (absolute difference        
      between predicted RGB and Ground Truth RGB).
    * Implement the SSIM Loss.
    * Add any GES-specific losses mentioned in the paper (e.g.,  
      Depth-SSIM or regularization terms to keep Surfels flat).  
  5. get_metrics_dict() & get_image_metrics_and_images():
    * Use torchmetrics for PSNR, SSIM, and LPIPS so you can      
      track training progress in the Nerfstudio viewer/WandB.    

Phase 4: GES-Specific Training Logic
Focus on the paper's custom training phases.

  6. get_training_callbacks():
    * Densification Hooks: Register the BEFORE_TRAIN_ITERATION   
      and AFTER_TRAIN_ITERATION callbacks. Inside the "after"    
      callback, you must call your strategy object(s) to
      split/clone/cull both your Surfels and Gaussians based on  
      their gradients.
    * The Discard Phase (Iter 10k): Write the logic to identify  
      and delete parameters based on the distance threshold (Eq. 
      7).
    * The Ramp Phase (Iter 18k-20k): Implement the logic that    
      modulates the $\tau$ parameter over these specific
      iterations to transition the blending.

Recommended Testing Strategy
Don't try to write all of this at once. I recommend this path:   
  1. Milestone 1: Write initialization and only the Surfel        
    rasterization pass. Ignore Gaussians entirely. Make sure it  
    trains like a normal 2D Gaussian splatting model.
  2. Milestone 2: Add the Gaussians and the second rasterization  
    pass, but use a simple addition for compositing.
  3. Milestone 3: Implement the complex depth-based blending      
    ($\tau$ modulation) between the two passes.
  4. Milestone 4: Add the specific callbacks (Discard, Ramp) to   
    finalize the paper's exact training schedule.