import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import random
import os
from PIL import Image # For displaying images
import tempfile # For creating temporary files reliably

# --- Constants (These MUST match your training script's parameters) ---
# If your training parameters change, update these accordingly
nc = 3       # Number of channels in the training images. For RGB, this is 3
nz = 100     # Size of z latent vector (i.e. size of generator input)
ngf = 64     # Size of feature maps in generator (e.g., 64)
ngpu = 1     # Number of GPUs available (for model loading on a single GPU)
image_size = 64 # Spatial size of generated images (e.g., 64x64)

# --- Generator Model Definition (MUST be identical to your training script's Generator) ---
# This class needs to be defined here so that Streamlit can instantiate the model
# before loading its state_dict.
class Generator(nn.Module):
    def __init__(self, ngpu_param):
        super(Generator, self).__init__() # Correct __init__ call
        self.ngpu = ngpu_param
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)

# --- Model Loading with Caching ---
# @st.cache_resource caches the function's return value.
# The model will be loaded only once when the app starts or when the checkpoint_path changes.
@st.cache_resource
def load_generator_model(checkpoint_path):
    # Determine the device for loading and running the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}") # This prints to the terminal/Colab output, not Streamlit app

    # Instantiate the Generator model
    loaded_netG = Generator(ngpu).to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- FIX START: Adjust state_dict keys if they have '_orig_mod.' prefix (from torch.compile) ---
    state_dict = checkpoint['netG_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            # Strip the prefix for keys from a compiled model
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            # Keep keys as is for non-compiled models (or if prefix isn't present)
            new_state_dict[k] = v
    # --- FIX END ---

    # Load the adjusted state dict into the model
    loaded_netG.load_state_dict(new_state_dict)
    loaded_netG.eval() # Set model to evaluation mode for inference

    # Apply torch.compile if PyTorch 2.x and GPU is available
    if torch.__version__.startswith("2.") and device.type == 'cuda':
        try:
            loaded_netG = torch.compile(loaded_netG, mode="reduce-overhead")
            print("Model compiled successfully with torch.compile for inference.")
        except Exception as e:
            print(f"Could not compile model: {e}")
            st.warning(f"Could not compile model with torch.compile: {e}. Running without compilation.")

    return loaded_netG, device

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="CelebA DCGAN Face Generator",
    page_icon="✨",
    layout="centered", # Can be "wide" for more space
    initial_sidebar_state="auto" # Can be "expanded" or "collapsed"
)

st.title("✨ CelebA DCGAN Face Generator ✨")
st.write("Upload your trained Generator model checkpoint (`.pth` file) to generate new celebrity-like faces!")
st.write("This application uses a DCGAN trained on the CelebA dataset.")

# --- Sidebar for Controls ---
st.sidebar.header("Model and Generation Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your Generator Checkpoint (.pth)",
    type="pth",
    help="Upload the `model_epoch_XXX.pth` file (e.g., `model_epoch_055.pth`) from your Google Drive output folder."
)

num_images_to_generate = st.sidebar.slider(
    "Number of Faces to Generate (Grid Size)",
    min_value=1,
    max_value=64, # Max 64 images for an 8x8 grid
    value=16, # Default to 4x4 grid (4x4 = 16 images)
    step=1,
    help="Choose how many faces to generate. Max 64 (for an 8x8 grid)."
)

generation_seed = st.sidebar.number_input(
    "Generation Seed (optional)",
    min_value=0,
    max_value=999999999,
    value=None, # Default to no seed (random)
    help="Enter a number for reproducible face generation. Leave blank for random generation each time."
)

# --- Main Content Area ---
if uploaded_file is not None:
    # Use tempfile to get a platform-agnostic temporary directory
    temp_dir = tempfile.gettempdir()
    # Create a path for the temporary checkpoint file
    temp_checkpoint_path = os.path.join(temp_dir, uploaded_file.name)

    netG = None # Initialize netG outside try block
    current_device = None # Initialize current_device outside try block

    try:
        with st.spinner("Loading model... This might take a moment if it's the first run or compiling."):
            # Write the uploaded file's content to the temporary file
            with open(temp_checkpoint_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the model using the cached function
            netG, current_device = load_generator_model(temp_checkpoint_path)
            st.success("Model loaded successfully!")

        st.write(f"Model will run on: `{current_device}`")

        # Button to trigger face generation
        if st.button("Generate Faces"):
            with st.spinner(f"Generating {num_images_to_generate} faces..."):
                # Set seed for reproducibility if provided
                if generation_seed is not None:
                    torch.manual_seed(generation_seed)
                    np.random.seed(generation_seed)
                    random.seed(generation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(generation_seed)
                    st.info(f"Using generation seed: `{generation_seed}`")
                else:
                    # Generate a random seed if none provided for transparency
                    current_seed = random.randint(0, 100000)
                    torch.manual_seed(current_seed)
                    np.random.seed(current_seed)
                    random.seed(current_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(current_seed)
                    st.info(f"Using random seed: `{current_seed}` (for this generation)")

                with torch.no_grad(): # Disable gradient calculations for inference
                    # Generate noise vector
                    noise = torch.randn(num_images_to_generate, nz, 1, 1, device=current_device)
                    # Generate images from the noise
                    generated_images_tensor = netG(noise).cpu() # Move to CPU for display

                # Create a grid of images for display
                # Ensure nrow is appropriate for the number of images to make a roughly square grid
                nrow = int(np.ceil(np.sqrt(num_images_to_generate)))
                grid = vutils.make_grid(generated_images_tensor, padding=2, normalize=True, nrow=nrow)

                # Convert PyTorch tensor (C, H, W) to NumPy array (H, W, C) for PIL/Streamlit display
                np_grid = np.transpose(grid.numpy(), (1, 2, 0))
                # Denormalize to 0-255 range and convert to uint8 type for image display
                np_grid = (np_grid * 255).astype(np.uint8)

                # Display the image grid in Streamlit
                st.image(np_grid, caption=f"{num_images_to_generate} Generated Faces", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred while loading or using the model: {e}")
        st.code(f"Error details: {e}", language='python') # Show error details for debugging
        st.warning("Please ensure you uploaded the correct `.pth` checkpoint file and that its architecture matches the `Generator` class defined in this script. Also check model constants like `nz`, `ngf`.")
    finally:
        # Clean up the temporary file, ensuring it's removed even if an error occurred
        if os.path.exists(temp_checkpoint_path):
            os.remove(temp_checkpoint_path)
            print(f"Cleaned up temporary file: {temp_checkpoint_path}") # This prints to terminal/Colab output

else:
    st.info("Please upload your trained Generator model checkpoint (`.pth` file) in the sidebar to get started.")
    st.markdown("""
    **To get your checkpoint file:**
    1.  Go to your Google Drive (`/content/drive/MyDrive/Colab_GAN_Output/CelebA_DCGAN_LocalData_v5/checkpoints/`)
    2.  Download the `model_epoch_XXX.pth` file (e.g., `model_epoch_055.pth`).
    3.  Upload it using the file uploader on the left sidebar.
    """)

st.markdown("---")
st.caption("Developed with ❤️ using PyTorch and Streamlit")