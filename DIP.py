import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.ndimage import convolve

# --- Page Configuration ---
st.set_page_config(
    page_title="DIP Exam Prep Assistant",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

def load_image(image_file):
    """Loads an image file and returns it as a PIL Image."""
    return Image.open(image_file)

def get_grayscale_image(img_array):
    """Converts a BGR or RGB image array to grayscale."""
    if len(img_array.shape) == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    return img_array

def create_sample_image():
    """Generates a sample noisy image if none is uploaded."""
    img = np.zeros((256, 256), dtype=np.uint8)
    # Create a white square in the middle
    img[64:192, 64:192] = 255
    # Add some salt and pepper noise
    noisy_img = random_noise(img, mode='s&p', amount=0.1)
    return (noisy_img * 255).astype(np.uint8)

def create_frequency_domain_image(img):
    """Computes the magnitude spectrum of an image for visualization."""
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum

# --- Main App ---

st.title("🎓 Digital Image Processing Interactive Study Guide")
st.markdown("Your personal assistant for the DIP End Semester Exam. All concepts are explained with theory, equations, and interactive examples.")

# --- Sidebar for Navigation ---
st.sidebar.title("📚 Exam Topics")
st.sidebar.markdown("Select a topic to study.")

topic_category = st.sidebar.radio(
    "Choose a category:",
    ('5 Mark Questions', '10 Mark Questions')
)

if topic_category == '5 Mark Questions':
    topic = st.sidebar.selectbox(
        "Select a 5-mark topic:",
        [
            "Adaptive Mean Filtering",
            "Hit-or-Miss Transformation",
            "Color Spaces (HSI)",
            "Laplacian of Gaussian (LoG)"
        ]
    )
else:
    topic = st.sidebar.selectbox(
        "Select a 10-mark topic:",
        [
            "Frequency Domain Filtering (Notch, Band-pass, Band-stop)",
            "Pattern Recognition by Minimum Distance Classifier",
            "Geometric Mean Filter in Frequency Domain"
        ]
    )

st.sidebar.markdown("---")
st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Upload your own image", type=["png", "jpg", "jpeg", "bmp"])

# --- Image Loading and Display ---
if uploaded_file is not None:
    image = load_image(uploaded_file)
    img_array = np.array(image)
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
else:
    st.sidebar.info("Using a default sample image. Upload an image for custom results.")
    img_array = create_sample_image()
    st.sidebar.image(img_array, caption='Default Sample Image', use_column_width=True)

gray_img = get_grayscale_image(img_array)

# ==============================================================================
# 5 MARK QUESTIONS
# ==============================================================================

if topic == "Adaptive Mean Filtering":
    st.header("1. Adaptive Mean Filtering for Local Noise Reduction")
    st.markdown("""
    Adaptive Mean Filtering is a sophisticated spatial filtering technique that performs better than a standard mean filter. Unlike the standard mean filter which operates uniformly across the entire image, the adaptive filter changes its behavior based on the statistical characteristics of the local neighborhood of each pixel.

    **Core Idea:** The filter's action depends on whether the local variance is high or low compared to the overall noise variance.
    - If the local variance is low (similar to noise variance), it means the area is relatively flat. The filter performs strong smoothing, similar to a mean filter.
    - If the local variance is high, it suggests the presence of edges or details. The filter performs less smoothing to preserve these features.
    """)

    st.subheader("Mathematical Formulation")
    st.latex(r'''
    \hat{f}(x, y) = g(x, y) - \frac{\sigma_\eta^2}{\sigma_L^2} \left( g(x, y) - \mu_L \right)
    ''')
    st.markdown(r"""
    Where:
    - $\hat{f}(x, y)$ is the estimated (filtered) pixel value at `(x, y)`.
    - $g(x, y)$ is the original (noisy) pixel value at `(x, y)`.
    - $\sigma_\eta^2$ is the variance of the noise in the entire image. This is often estimated beforehand.
    - $\mu_L$ is the local mean in a neighborhood `S_xy` around `(x, y)`.
    - $\sigma_L^2$ is the local variance in the neighborhood `S_xy`.

    **Two main cases arise from this equation:**
    1.  **If $\sigma_\eta^2$ is zero (no noise):** The fraction becomes zero, and $\hat{f}(x, y) = g(x, y)$. The filter does nothing, preserving the original image.
    2.  **If $\sigma_L^2 \approx \sigma_\eta^2$:** The fraction is close to 1, so $\hat{f}(x, y) \approx \mu_L$. This is strong smoothing, as seen in a standard mean filter.
    3.  **If $\sigma_L^2 \gg \sigma_\eta^2$:** The fraction is close to 0, so $\hat{f}(x, y) \approx g(x, y)$. This preserves edges and details where local variance is high.
    """)

    st.subheader("Interactive Demo")
    col1, col2 = st.columns(2)

    with col1:
        st.image(gray_img, caption="Original Grayscale Image", use_column_width=True)

    with col2:
        kernel_size = st.slider("Select neighborhood size (must be odd)", 3, 21, 5, 2)
        # Estimate noise variance from a flat area (e.g., top-left corner)
        noise_var_est = np.var(gray_img[:20, :20].astype(float))
        st.write(f"Estimated Noise Variance ($\sigma_\\eta^2$): {noise_var_est:.2f}")

        # Applying the adaptive filter
        filtered_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, kernel_size, 2) # This is a simplification
        # A more accurate implementation:
        local_mean = cv2.blur(gray_img.astype(float), (kernel_size, kernel_size))
        local_var = cv2.blur(gray_img.astype(float)**2, (kernel_size, kernel_size)) - local_mean**2
        
        noise_var = max(noise_var_est, 1e-5) # Avoid division by zero
        local_var = np.maximum(local_var, noise_var) # Ensure local_var is not smaller than noise_var

        ratio = noise_var / local_var
        filtered_img_correct = gray_img - ratio * (gray_img - local_mean)
        filtered_img_correct = np.clip(filtered_img_correct, 0, 255).astype(np.uint8)

        st.image(filtered_img_correct, caption=f"Adaptive Mean Filtered (Kernel: {kernel_size}x{kernel_size})", use_column_width=True)

    st.success("**Key Takeaway:** Adaptive mean filtering is superior to standard mean filtering because it preserves edges while smoothing noise, by adapting its behavior to local image statistics.")

elif topic == "Hit-or-Miss Transformation":
    st.header("2. Hit-or-Miss Transformation for Shape Detection")
    st.markdown("""
    The Hit-or-Miss Transform is a fundamental morphological operation used for finding specific patterns (shapes) in a binary image. It's a powerful tool for shape detection and is the basis for other advanced morphological algorithms like thinning and pruning.

    **Core Idea:** The transform uses a pair of structuring elements (SEs) to simultaneously probe the foreground and background of an image.
    1.  **The "Hit" SE ($B_1$):** This element must fit entirely within the foreground object at a given position. It looks for the shape of the object itself.
    2.  **The "Miss" SE ($B_2$):** This element must fit entirely within the background around the object. It looks for the shape of the surrounding background.

    A "match" occurs at a pixel `(x, y)` only if the "Hit" SE fits the foreground AND the "Miss" SE fits the background at that exact location.
    """)

    st.subheader("Mathematical Formulation")
    st.markdown("The Hit-or-Miss transform of an image `A` by a composite structuring element `B = (B1, B2)` is defined as the set of all points `x` where `B1` translated to `x` fits in `A`, and `B2` translated to `x` fits in the complement of `A` ($A^c$).")
    st.latex(r'''
    A \circledast B = (A \ominus B_1) \cap (A^c \ominus B_2)
    ''')
    st.markdown(r"""
    Where:
    - $A \circledast B$ is the Hit-or-Miss operation.
    - $A \ominus B_1$ is the erosion of `A` by the "Hit" structuring element $B_1$. This finds all pixels where $B_1$ can fit.
    - $A^c \ominus B_2$ is the erosion of the complement of `A` (the background) by the "Miss" structuring element $B_2$. This finds all pixels where $B_2$ fits in the background.
    - $\cap$ is the set intersection, meaning we only keep the pixels that satisfy both conditions.
    """)

    st.subheader("Interactive Demo")
    st.markdown("Let's try to find corners in the image. We'll use a structuring element that looks for a specific corner shape.")

    # Binarize the image for morphological operations
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    col1, col2 = st.columns(2)
    with col1:
        st.image(binary_img, caption="Original Binary Image", use_column_width=True)

    with col2:
        # Define a structuring element to find top-left corners
        hit_kernel = np.array([[0, 1, 1],
                               [0, 1, 0],
                               [0, 0, 0]], dtype=np.uint8)
        miss_kernel = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [1, 1, 0]], dtype=np.uint8)
        
        # In OpenCV, the hit-or-miss kernel is combined. 1 for hit, -1 for miss, 0 for don't care.
        combined_kernel = np.array([[0, 1, 1],
                                    [-1, 1, 0],
                                    [-1, -1, 0]], dtype=np.int8)

        hit_or_miss_result = cv2.morphologyEx(binary_img, cv2.MORPH_HITMISS, combined_kernel)

        st.image(hit_or_miss_result, caption="Hit-or-Miss Result (Detecting Top-Right Corners)", use_column_width=True)
        st.info("The white pixels in the result mark the locations where the specific corner shape was found.")

    st.success("**Key Takeaway:** The Hit-or-Miss transform is a precise shape detector in binary images, using a pair of structuring elements to match both the object's shape and its immediate background.")


elif topic == "Color Spaces (HSI)":
    st.header("3. Color Spaces (HSI Description)")
    st.markdown("""
    Computers typically represent color using the **RGB (Red, Green, Blue)** model, which is an additive model well-suited for displays. However, the RGB model is not intuitive for human perception. We don't describe a color by its R, G, and B values; we describe it by its tint, shade, and brightness.

    The **HSI (Hue, Saturation, Intensity)** color model decouples the color information from the intensity (brightness) information, which aligns much better with how humans perceive color. This makes it extremely useful for many image processing tasks.
    """)

    st.subheader("Fundamental Description of HSI Components")
    st.markdown("""
    - **Hue (H):** This represents the pure color itself (e.g., red, yellow, green, blue). It is measured as an angle from 0 to 360 degrees on a color wheel. Red is typically at 0°, Green at 120°, and Blue at 240°.
    - **Saturation (S):** This represents the "purity" or "vibrancy" of the color. It indicates how much white light is mixed with the pure hue. A saturation of 1 means a pure, vivid color. A saturation of 0 means a shade of gray (achromatic).
    - **Intensity (I):** This represents the brightness or luminance of the color. It is a measure of how light or dark the color is, ranging from black (0) to white (1). This component is decoupled from the color information.

    **Why is this useful?** By separating intensity from hue and saturation, we can perform processing on the brightness of an image without altering its color content. For example, applying histogram equalization to the 'I' channel of an HSI image will improve its contrast without changing its colors, which is not possible in the RGB space directly.
    """)

    st.subheader("Visualization of HSI Components")
    if len(img_array.shape) < 3:
        st.warning("HSI requires a color image. Please upload a color image to see the HSI components.")
    else:
        # Ensure image is RGB for conversion
        if img_array.shape[2] == 4: # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] == 1: # Grayscale
             st.warning("HSI requires a color image.")
             st.stop()
        
        # Convert BGR (from OpenCV) to RGB for display
        rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        hsi_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS) # OpenCV uses HLS which is very similar to HSI

        h, l, s = cv2.split(hsi_img) # Note: OpenCV's HLS order is H, L, S

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(rgb_img, caption="Original RGB Image", use_column_width=True)
        with col2:
            st.image(h, caption="Hue (H) Channel", use_column_width=True)
        with col3:
            st.image(s, caption="Saturation (S) Channel", use_column_width=True)
        with col4:
            st.image(l, caption="Intensity/Lightness (I/L) Channel", use_column_width=True)

        st.markdown("Notice how the **Intensity** channel looks like a grayscale version of the original image, while the **Hue** and **Saturation** channels capture the color information.")

    st.success("**Key Takeaway:** The HSI color model is a perception-based model that separates color information (Hue, Saturation) from brightness (Intensity), making it ideal for tasks where you want to modify image brightness without affecting its colors.")


elif topic == "Laplacian of Gaussian (LoG)":
    st.header("4. Laplacian of Gaussian (LoG)")
    st.markdown("""
    The Laplacian of Gaussian (LoG) is a powerful edge detection operator. It combines two steps into one operation to find edges, particularly zero-crossings, which correspond to the locations of edges in an image.

    The two steps are:
    1.  **Gaussian Smoothing:** First, the image is smoothed with a Gaussian filter. This is done to reduce noise, as the Laplacian operator is very sensitive to noise.
    2.  **Laplacian Operator:** Second, the Laplacian operator is applied to the smoothed image. The Laplacian is a 2nd-order derivative operator that highlights regions of rapid intensity change.

    Edges in the LoG-filtered image are found at the **zero-crossings** – the points where the pixel values cross from positive to negative or vice-versa.
    """)

    st.subheader("The 'Mexican Hat' Function")
    st.markdown("""
    When you visualize the LoG kernel in 3D, it looks like a sombrero, which is why it's often called the **Mexican Hat** function.
    """)

    st.subheader("Mathematical Formulation")
    st.latex(r'''
    \nabla^2 G(x, y) = \frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2}
    ''')
    st.markdown("Where $G(x,y)$ is the 2D Gaussian function:")
    st.latex(r'''
    G(x, y) = e^{-\frac{x^2 + y^2}{2\sigma^2}}
    ''')
    st.markdown("Combining these gives the LoG function:")
    st.latex(r'''
    \text{LoG}(x, y) = -\frac{1}{\pi\sigma^4} \left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) e^{-\frac{x^2 + y^2}{2\sigma^2}}
    ''')

    st.error("""
    **Why is there a negative sign in the LoG formula?**

    The negative sign is included by convention to turn the central peak of the "Mexican Hat" positive.
    - The standard 2D Gaussian function has a positive peak at the center.
    - The Laplacian operator, being a second derivative, measures the "concavity" of a function. At the peak of the Gaussian, the concavity is negative (it's curving downwards).
    - Therefore, the raw Laplacian of a Gaussian results in a kernel with a negative value at its center and positive values in the surrounding ring.
    - By **adding a negative sign** to the entire formula, we invert this, creating the familiar "Mexican Hat" shape with a **positive central peak and a negative surrounding ring**. This makes the kernel's visual representation more intuitive.
    """)

    st.subheader("Interactive Demo")
    col1, col2 = st.columns(2)
    with col1:
        st.image(gray_img, caption="Original Grayscale Image", use_column_width=True)

    with col2:
        kernel_size = st.slider("Select Gaussian kernel size (odd)", 3, 15, 5, 2)
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
        # Apply Laplacian
        log_img = cv2.Laplacian(blurred_img, cv2.CV_64F)
        # We take the absolute value for visualization, but zero-crossing detection is the proper way to find edges.
        log_img_display = cv2.convertScaleAbs(log_img)
        st.image(log_img_display, caption=f"LoG Filtered Image (Kernel: {kernel_size}x{kernel_size})", use_column_width=True)
        st.info("The bright lines represent areas of high second-derivative, corresponding to edges.")

    st.success("**Key Takeaway:** LoG is a robust edge detector that first smooths the image to reduce noise and then uses the Laplacian to find edges at zero-crossings. The negative sign in its formula is a convention to make the central peak of its kernel positive.")


# ==============================================================================
# 10 MARK QUESTIONS
# ==============================================================================

elif topic == "Frequency Domain Filtering (Notch, Band-pass, Band-stop)":
    st.header("1. Frequency Domain Filtering")
    st.markdown("""
    Frequency domain filtering is a powerful technique where we modify the Fourier Transform of an image to achieve filtering effects. The core process is:
    1.  Compute the 2D Discrete Fourier Transform (DFT) of the input image.
    2.  Shift the zero-frequency component to the center for easier visualization and filter design.
    3.  Create a filter mask `H(u, v)` of the same size as the image.
    4.  Multiply the shifted DFT pointwise with the filter mask: `G(u, v) = F(u, v) * H(u, v)`.
    5.  Shift the result back.
    6.  Compute the Inverse DFT to get the filtered image back in the spatial domain.

    Low frequencies correspond to the general, smooth areas of an image, while high frequencies correspond to details, edges, and noise.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image(gray_img, caption="Original Grayscale Image", use_column_width=True)
    with col2:
        magnitude_spectrum = create_frequency_domain_image(gray_img)
        st.image(magnitude_spectrum, caption="Magnitude Spectrum (Frequency Domain)", use_column_width=True)

    st.subheader("Filter Types")
    filter_type = st.radio("Select Filter Type to Apply:", ("Notch Filter", "Band-stop Filter", "Band-pass Filter"))

    rows, cols = gray_img.shape
    crow, ccol = rows // 2, cols // 2

    # Create a base mask
    mask = np.ones((rows, cols), np.uint8)
    
    if filter_type == "Notch Filter":
        st.markdown("""
        **Notch Filter:** A notch filter rejects (sets to zero) frequencies in a specific, small region of the frequency domain. It is primarily used to remove periodic noise, which appears as bright spots (spikes) in the magnitude spectrum.
        - **Action:** Attenuates a predefined neighborhood around a specific frequency location.
        - **Use Case:** Removing sinusoidal noise, moiré patterns, or interference from a power line (e.g., 60 Hz hum).
        """)
        st.info("Interactive Demo: We will simulate removing periodic noise by placing notches.")
        
        # Notch parameters
        u0 = st.slider("Notch Center u-coordinate (from center)", -crow, crow, 30)
        v0 = st.slider("Notch Center v-coordinate (from center)", -ccol, ccol, 30)
        radius = st.slider("Notch Radius", 1, 50, 10)

        # Create notch
        cv2.circle(mask, (ccol + v0, crow + u0), radius, 0, -1)
        # Create symmetric notch
        cv2.circle(mask, (ccol - v0, crow - u0), radius, 0, -1)
        
    elif filter_type == "Band-stop Filter":
        st.markdown("""
        **Band-stop Filter (or Band-reject):** This filter attenuates frequencies within a certain range (a band) from the center of the spectrum. It's like a more general version of a low-pass or high-pass filter.
        - **Action:** Passes low and high frequencies but rejects frequencies in a ring-shaped band.
        - **Use Case:** Removing specific frequency-band noise while preserving the very low and very high frequencies.
        """)
        st.info("Interactive Demo: We will create a ring that blocks a band of frequencies.")
        
        r_outer = st.slider("Outer Radius of Stop Band (D0 + W/2)", 1, crow, 80)
        r_inner = st.slider("Inner Radius of Stop Band (D0 - W/2)", 0, crow-1, 60)

        # Create the ring
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2
        band_mask = (mask_area >= r_inner**2) & (mask_area <= r_outer**2)
        mask[band_mask] = 0

    elif filter_type == "Band-pass Filter":
        st.markdown("""
        **Band-pass Filter:** This is the opposite of a band-stop filter. It passes frequencies within a certain band and rejects all others (both low and high).
        - **Action:** Rejects low and high frequencies but passes frequencies in a ring-shaped band.
        - **Use Case:** Isolating features of a specific size or texture, as different textures often correspond to specific frequency bands.
        """)
        st.info("Interactive Demo: We will create a ring that allows a band of frequencies to pass.")
        
        r_outer = st.slider("Outer Radius of Pass Band (D0 + W/2)", 1, crow, 80)
        r_inner = st.slider("Inner Radius of Pass Band (D0 - W/2)", 0, crow-1, 60)
        
        mask = np.zeros((rows, cols), np.uint8)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2
        band_mask = (mask_area >= r_inner**2) & (mask_area <= r_outer**2)
        mask[band_mask] = 1

    # --- Apply the selected filter ---
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Apply mask to both real and imaginary parts
    fshift_filtered = dft_shift.copy()
    fshift_filtered[:, :, 0] = dft_shift[:, :, 0] * mask
    fshift_filtered[:, :, 1] = dft_shift[:, :, 1] * mask

    # Inverse transform
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    col3, col4 = st.columns(2)
    with col3:
        st.image(mask * 255, caption="Filter Mask H(u, v) (Black=0, White=1)", use_column_width=True)
    with col4:
        st.image(img_back, caption="Filtered Image", use_column_width=True)

    st.success("**Key Takeaway:** Frequency domain filters provide precise control over which image frequencies are kept or discarded. Notch filters target specific noise points, while band-stop/band-pass filters work on entire frequency ranges.")


elif topic == "Pattern Recognition by Minimum Distance Classifier":
    st.header("2. Pattern Recognition by Minimum Distance Classifier")
    st.markdown("""
    The Minimum Distance Classifier is one of the simplest yet most fundamental classifiers in pattern recognition. It classifies an unknown pattern (represented by a feature vector) into a class based on the distance to the mean (or prototype) of each class.

    **Core Idea:**
    1.  **Training Phase:** For each class, we compute a "prototype" vector, which is typically the mean of all training samples belonging to that class.
    2.  **Classification Phase:** For a new, unknown sample, we compute its feature vector. Then, we calculate the distance from this new vector to the prototype vector of each class.
    3.  **Decision:** The new sample is assigned to the class whose prototype is closest (i.e., has the minimum distance).

    The "distance" is usually the **Euclidean distance**.
    """)

    st.subheader("Mathematical Formulation")
    st.markdown(r"""
    Let's say we have `M` classes, $\omega_1, \omega_2, \dots, \omega_M$.
    The mean (prototype) vector for each class $\omega_j$ is $\mathbf{m}_j$:
    """)
    st.latex(r'''
    \mathbf{m}_j = \frac{1}{N_j} \sum_{\mathbf{x} \in \omega_j} \mathbf{x}
    ''')
    st.markdown(r"""
    Where $N_j$ is the number of training vectors in class $\omega_j$.

    To classify an unknown vector $\mathbf{x}$, we compute the Euclidean distance $D_j(\mathbf{x})$ to each mean vector $\mathbf{m}_j$:
    """)
    st.latex(r'''
    D_j(\mathbf{x}) = \| \mathbf{x} - \mathbf{m}_j \| = \sqrt{(\mathbf{x} - \mathbf{m}_j)^T (\mathbf{x} - \mathbf{m}_j)}
    ''')
    st.markdown(r"""
    The decision rule is: Assign $\mathbf{x}$ to class $\omega_i$ if $D_i(\mathbf{x}) < D_j(\mathbf{x})$ for all $j \neq i$.
    
    This creates linear decision boundaries between classes.
    """)

    st.subheader("Interactive Demo: Classifying Pixels by Color")
    st.markdown("""
    Let's use the Minimum Distance Classifier to segment an image based on color. We will define three classes: **Reddish**, **Greenish**, and **Bluish**.
    - The **feature vector** for each pixel will be its `(R, G, B)` values.
    - The **class prototypes** will be pure Red `(255, 0, 0)`, Green `(0, 255, 0)`, and Blue `(0, 0, 255)`.
    - Each pixel in the image will be classified into the color class its feature vector is closest to.
    """)

    if len(img_array.shape) < 3:
        st.warning("This demo requires a color image. Please upload one.")
    else:
        # Ensure image is RGB
        if img_array.shape[2] == 4:
            rgb_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_img, caption="Original Color Image", use_column_width=True)

        # Define class prototypes (means)
        class_means = {
            "Reddish": np.array([255, 0, 0]),
            "Greenish": np.array([0, 255, 0]),
            "Bluish": np.array([0, 0, 255])
        }
        
        # Reshape image to be a list of pixels (feature vectors)
        pixels = rgb_img.reshape(-1, 3).astype(np.float32)
        
        # Calculate distances for all pixels to each class mean
        dist_red = np.linalg.norm(pixels - class_means["Reddish"], axis=1)
        dist_green = np.linalg.norm(pixels - class_means["Greenish"], axis=1)
        dist_blue = np.linalg.norm(pixels - class_means["Bluish"], axis=1)
        
        # Stack distances and find the index of the minimum distance for each pixel
        distances = np.vstack([dist_red, dist_green, dist_blue])
        labels = np.argmin(distances, axis=0)
        
        # Create the segmented output image
        segmented_img = np.zeros_like(pixels, dtype=np.uint8)
        segmented_img[labels == 0] = class_means["Reddish"]
        segmented_img[labels == 1] = class_means["Greenish"]
        segmented_img[labels == 2] = class_means["Bluish"]
        
        segmented_img = segmented_img.reshape(rgb_img.shape)

        with col2:
            st.image(segmented_img, caption="Segmented by Minimum Distance to R, G, B", use_column_width=True)

    st.success("**Key Takeaway:** The Minimum Distance Classifier provides a simple and fast way to partition feature space using linear decision boundaries. It's effective when classes are well-separated and have a compact, spherical distribution.")


elif topic == "Geometric Mean Filter in Frequency Domain":
    st.header("3. Geometric Mean Filter in Frequency Domain")
    st.markdown("""
    The Geometric Mean filter is a non-linear spatial filter that is effective at removing Gaussian noise while preserving edges better than the Arithmetic Mean filter. Its spatial domain formula is:
    """)
    st.latex(r'''
    \hat{f}(x, y) = \left[ \prod_{(s, t) \in S_{xy}} g(s, t) \right]^{\frac{1}{mn}}
    ''')
    st.markdown(r"""
    Where $S_{xy}$ is the neighborhood of size $m \times n$ centered at $(x, y)$, and $g(s, t)$ are the pixel values in that neighborhood.

    **Challenge:** The product operation ($\prod$) is computationally intensive. Can we implement this in the frequency domain?

    **Yes, by using logarithms.** The logarithm of a product is the sum of logarithms:
    """)
    st.latex(r'''
    \log[\hat{f}(x, y)] = \frac{1}{mn} \sum_{(s, t) \in S_{xy}} \log[g(s, t)]
    ''')
    st.markdown("""
    This equation is now a convolution! The term on the right is simply the **arithmetic mean** of the **logarithm of the image**. We can perform arithmetic mean filtering (which is a convolution with a box kernel) in the frequency domain.

    ### The Frequency Domain Process
    1.  **Log Transform:** Take the natural logarithm of the input image `g(x, y)`. Let's call this `L(x, y) = log[g(x, y)]`. (We must add 1 to the image to avoid `log(0)`).
    2.  **Frequency Domain Convolution:**
        a. Compute the DFT of `L(x, y)`.
        b. Create a frequency-domain representation of the arithmetic mean filter (which is a low-pass filter, e.g., a Butterworth or Gaussian LPF).
        c. Multiply the DFT of `L` with the filter.
    3.  **Inverse DFT:** Compute the inverse DFT of the result to get the smoothed log image, `L_smooth(x, y)`.
    4.  **Exponentiation:** Take the exponent of the result to reverse the initial log transform: `f_hat(x, y) = exp[L_smooth(x, y)]`.

    This complex process achieves the geometric mean filtering effect using the efficiency of frequency domain convolution.
    """)

    st.subheader("Interactive Demo")
    # Add some Gaussian noise to the image for a better demo
    noisy_img = random_noise(gray_img, mode='gaussian', var=0.01)
    noisy_img = (255 * noisy_img).astype(np.uint8)

    col1, col2 = st.columns(2)
    with col1:
        st.image(noisy_img, caption="Original Image with Gaussian Noise", use_column_width=True)

    # --- Frequency Domain Geometric Mean ---
    # 1. Log Transform
    log_img = np.log1p(noisy_img.astype(np.float32))

    # 2. Frequency Domain Convolution (using a Gaussian LPF as the mean filter)
    d0 = st.slider("Cutoff Frequency (D0) for Mean Filter", 1, 100, 30)
    
    rows, cols = log_img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Gaussian LPF mask
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2
    glpf_mask = np.exp(-(mask_area / (2 * d0**2)))

    dft = cv2.dft(log_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    fshift_filtered = dft_shift.copy()
    fshift_filtered[:, :, 0] = dft_shift[:, :, 0] * glpf_mask
    fshift_filtered[:, :, 1] = dft_shift[:, :, 1] * glpf_mask

    # 3. Inverse DFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    log_smooth = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 4. Exponentiation
    geo_mean_img = np.expm1(log_smooth)
    geo_mean_img = cv2.normalize(geo_mean_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    with col2:
        st.image(geo_mean_img, caption=f"Geometric Mean Filtered (Freq. Domain, D0={d0})", use_column_width=True)

    st.success("**Key Takeaway:** The geometric mean filter can be implemented in the frequency domain by transforming the problem into an arithmetic mean (convolution) problem using logarithms. This involves a log-transform, frequency-domain low-pass filtering, and an inverse exponentiation step.")

