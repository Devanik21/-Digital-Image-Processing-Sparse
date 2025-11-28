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
    ### Introduction: The Limitations of Standard Mean Filtering
    Before diving into the adaptive mean filter, it's crucial to understand why it was developed. The standard **Arithmetic Mean Filter** is a simple and intuitive method for noise reduction. It operates by replacing the value of each pixel with the average value of its neighbors within a defined kernel (e.g., a 3x3 or 5x5 window). While effective at smoothing out random noise like Gaussian noise, it has a significant drawback: **it blurs the entire image indiscriminately.**

    This uniform blurring occurs because the mean filter does not distinguish between noise and important image features like edges, lines, or fine textures. When the filter's kernel passes over an edge, it averages the high-intensity values on one side of the edge with the low-intensity values on the other, resulting in a smeared, less-defined edge. For many applications, especially in medical imaging or object recognition, preserving these details is paramount.

    ### The Adaptive Approach: Intelligence in Filtering
    **Adaptive Mean Filtering** is a more sophisticated spatial filtering technique designed to overcome this limitation. The key word here is "adaptive." Unlike the standard mean filter, which applies the same logic across the entire image, the adaptive filter **changes its behavior based on the statistical characteristics of the local neighborhood of each pixel.**

    **Core Idea:** The filter's action is governed by a comparison between the local image statistics within a sliding window and the overall noise characteristics of the image. Specifically, it analyzes the **local mean ($\mu_L$)** and **local variance ($\sigma_L^2$)** against the **global noise variance ($\sigma_\eta^2$)**.

    -   **In flat, noisy areas:** The local variance will be low and roughly equal to the noise variance. In this case, the filter concludes it's looking at a region of pure noise and applies strong smoothing by replacing the pixel with the local mean.
    -   **In areas with details (edges, lines):** The local variance will be high, significantly greater than the noise variance. The filter recognizes this as a region containing important image features and applies minimal or no smoothing, thereby preserving the original pixel value and keeping the edge sharp.
    """)

    st.subheader("Mathematical Formulation")
    st.latex(r'''
    \hat{f}(x, y) = g(x, y) - \frac{\sigma_\eta^2}{\sigma_L^2} \left( g(x, y) - \mu_L \right)
    ''')
    st.markdown(r"""
    Let's break down each component of this powerful equation:

    -   **$\hat{f}(x, y)$**: This is the **output**, the new, filtered pixel value at coordinates `(x, y)`.
    -   **$g(x, y)$**: This is the **input**, the original (noisy) pixel value at `(x, y)`.
    -   **$\sigma_\eta^2$**: This is the **global noise variance** for the entire image. It represents the overall level of noise we expect. A critical point is that this value is often *unknown* and must be estimated. A common method is to select a small, relatively flat patch of the image and calculate its variance, assuming that any variation in that patch is due to noise.
    -   **$\mu_L$**: This is the **local mean** calculated within a neighborhood window `S_xy` (e.g., 5x5) centered on the pixel `(x, y)`. It's the average pixel value in that window.
    -   **$\sigma_L^2$**: This is the **local variance** within the same neighborhood `S_xy`. It measures how much the pixel values within the window vary from the local mean. A high local variance indicates the presence of edges or texture, while a low local variance indicates a flat region.

    ### Deep Dive into the Filter's Behavior

    The behavior of the filter is dictated by the ratio $\frac{\sigma_\eta^2}{\sigma_L^2}$. Let's analyze the three key scenarios that arise from this equation:

    1.  **Case 1: Noiseless Image ($\sigma_\eta^2 = 0$)**
        If the global noise variance is zero, it implies the image is perfectly clean. The ratio $\frac{0}{\sigma_L^2}$ becomes 0. The equation simplifies to:
        $\hat{f}(x, y) = g(x, y) - 0 \cdot (g(x, y) - \mu_L) = g(x, y)$.
        **Result:** The filter returns the original pixel value. This is the desired behavior; a good filter should not alter a clean image.

    2.  **Case 2: High Local Variance ($\sigma_L^2 \gg \sigma_\eta^2$)**
        This occurs when the filter window is over an edge or a highly textured area. The local variation is dominated by the image content, not the noise. In this case, the ratio $\frac{\sigma_\eta^2}{\sigma_L^2}$ approaches 0. The equation again simplifies to:
        $\hat{f}(x, y) \approx g(x, y)$.
        **Result:** The filter returns a value very close to the original pixel value. This is the "edge-preserving" property. The filter intelligently recognizes the detail and avoids blurring it.

    3.  **Case 3: Low Local Variance ($\sigma_L^2 \approx \sigma_\eta^2$)**
        This happens when the filter window is over a flat, uniform region where the only variations are due to noise. The local variance is similar to the global noise variance. The ratio $\frac{\sigma_\eta^2}{\sigma_L^2}$ is approximately 1. The equation becomes:
        $\hat{f}(x, y) \approx g(x, y) - 1 \cdot (g(x, y) - \mu_L) = g(x, y) - g(x, y) + \mu_L = \mu_L$.
        **Result:** The filter returns the local mean. This is equivalent to the action of a standard arithmetic mean filter, providing strong smoothing to eliminate the noise in the flat region.

    A final consideration is when $\sigma_L^2 < \sigma_\eta^2$. To prevent instability and over-subtraction, the implementation ensures that the ratio $\frac{\sigma_\eta^2}{\sigma_L^2}$ does not exceed 1 by setting $\sigma_L^2 = \max(\sigma_L^2, \sigma_\eta^2)$.

    ### Advantages and Disadvantages

    **Advantages:**
    -   **Superior Edge Preservation:** Its primary advantage over the standard mean filter is its ability to smooth noise without significantly blurring important image details.
    -   **Effective for Additive Noise:** It performs particularly well for removing additive Gaussian noise.
    -   **Conceptually Sound:** The logic is based on local statistics, making it an intuitive and powerful approach.

    **Disadvantages:**
    -   **Dependence on Noise Estimation:** The filter's performance is highly dependent on an accurate estimate of the global noise variance ($\sigma_\eta^2$). An incorrect estimate can lead to either insufficient smoothing or excessive blurring.
    -   **Computational Cost:** It is more computationally expensive than a simple mean filter because it requires calculating both the local mean and local variance for every pixel in the image.
    -   **Ineffective for Impulsive Noise:** It is not the best choice for removing "salt-and-pepper" (impulsive) noise. The extreme values of salt (255) and pepper (0) can heavily skew the local mean and variance calculations. For such noise, a **median filter** is generally superior.
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
    st.header("2. Hit-or-Miss Transformation for Precise Shape Detection")
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
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    col1, col2 = st.columns(2)
    with col1:
        st.image(binary_img, caption="Original Binary Image", use_column_width=True)

    with col2:
        # Define a structuring element to find top-left corners
        st.markdown("""
        **Defining the Structuring Element (SE):**
        To find a specific shape, we need a composite SE. In OpenCV, this is a single kernel where:
        - **`1`** represents a "Hit" pixel (must be part of the foreground).
        - **`-1`** represents a "Miss" pixel (must be part of the background).
        - **`0`** represents a "Don't Care" pixel.
        
        Let's design a kernel to find **top-right corners**:
        ```
        [[ -1, -1,  0 ],   // Background, Background, Don't Care
         [  1,  1,  0 ],   // Foreground, Foreground, Don't Care
         [ -1,  1,  0 ]]   // Background, Foreground, Don't Care
        ```
        This kernel looks for a pattern where there's a horizontal foreground line with a pixel below it, and background pixels to the top-left and bottom-left.
        """)

        kernel_choice = st.selectbox(
            "Select the shape to detect:",
            ("Top-Right Corner", "Bottom-Left Corner", "Isolated Pixel", "Line Ending (Right)")
        )

        if kernel_choice == "Top-Right Corner":
            # This kernel is designed to find corners like the top-right of a square.
            # It requires background pixels (-1) to the top and left of the corner,
            # and foreground pixels (1) forming the corner itself.
            combined_kernel = np.array([
                [-1, 1, 0],
                [-1, 1, 1],
                [-1, 1, 0]], dtype=np.int8)
            caption_text = "Hit-or-Miss: Detecting Top-Right Corners"
        elif kernel_choice == "Bottom-Left Corner":
            # The inverse of the top-right corner.
            combined_kernel = np.array([
                [0, 1, -1],
                [1, 1, -1],
                [0, 1, -1]], dtype=np.int8)
            caption_text = "Hit-or-Miss: Detecting Bottom-Left Corners"
        elif kernel_choice == "Isolated Pixel":
            # This kernel finds single foreground pixels that are completely surrounded by background.
            # The center is the "Hit" (1), and all its immediate neighbors are "Misses" (-1).
            combined_kernel = np.array([
                [-1, -1, -1],
                [-1,  1, -1],
                [-1, -1, -1]], dtype=np.int8)
            caption_text = "Hit-or-Miss: Detecting Isolated Pixels"
        else: # Line Ending (Right)
            # This kernel finds the rightmost end of a horizontal line.
            # It requires a foreground pixel (1) to its left and background pixels (-1)
            # above, below, and to its right.
            combined_kernel = np.array([
                [-1, -1, -1],
                [ 1,  1, -1],
                [-1, -1, -1]], dtype=np.int8)
            caption_text = "Hit-or-Miss: Detecting Right End of a Line"

        hit_or_miss_result = cv2.morphologyEx(binary_img, cv2.MORPH_HITMISS, combined_kernel)

        st.image(hit_or_miss_result, caption=caption_text, use_column_width=True)
        st.info("The white pixels in the result mark the locations where the specific corner shape was found.")

    st.markdown("""
    ### Applications and Relationship to Other Operations
    The Hit-or-Miss transform is more than just a standalone shape detector; it's a building block for other critical morphological algorithms.
    - **Thinning:** Thinning is an operation that erodes away the boundary of a foreground object while preserving its overall shape and connectivity. It is performed by iteratively applying the Hit-or-Miss transform with a sequence of structuring elements that detect and remove boundary pixels, until no more pixels can be removed. This is used to compute the "skeleton" of an object.
    - **Thickening:** The dual of thinning, thickening grows selected foreground pixels. It is also based on the Hit-or-Miss transform, but instead of removing the matched pixels, it sets them to the foreground color.
    - **Pruning:** This is used to clean up skeletonized images by removing small, parasitic branches. This can be achieved by using Hit-or-Miss to find line endings, dilating the result to cover the small branch, and then subtracting this from the original skeleton.
    - **Template Matching:** At its core, the operation is a form of binary template matching, finding exact instances of a pattern.
    """)

    st.success("**Key Takeaway:** The Hit-or-Miss transform is a highly specific shape detector for binary images. It uniquely uses a pair of structuring elements to match both the foreground (hit) and background (miss) simultaneously, making it the foundation for advanced morphological operations like thinning and pruning.")


elif topic == "Color Spaces (HSI)":
    st.header("3. Color Spaces (HSI Description)")
    st.markdown("""
    Computers typically represent color using the **RGB (Red, Green, Blue)** model, which is an additive model well-suited for displays. However, the RGB model is not intuitive for human perception. We don't describe a color by its R, G, and B values; we describe it by its tint, shade, and brightness.

    The **HSI (Hue, Saturation, Intensity)** color model decouples the color information from the intensity (brightness) information, which aligns much better with how humans perceive color. This makes it extremely useful for many image processing tasks.

    ### The Problem with the RGB Model
    In the RGB model, color and brightness are highly correlated. For example, to make a color brighter, you need to increase the values of R, G, and B. To change the color, you need to change the *ratio* of R, G, and B. This entanglement makes seemingly simple tasks difficult. For instance, if you want to improve the contrast of a color image using histogram equalization, applying it to the R, G, and B channels separately will not only change the contrast but also drastically and unnaturally alter the colors. This is because the relative proportions of R, G, and B are changed independently, leading to a phenomenon known as "color shifting."

    The HSI model solves this by providing an orthogonal representation of color and brightness, allowing us to manipulate one without affecting the other.
    """)

    st.subheader("Fundamental Description of HSI Components")
    st.markdown("""
    Imagine a cylindrical or conical color space. The HSI components map to this space in a very intuitive way:

    -   **Hue (H):** This represents the pure, dominant wavelength of the color (e.g., 'red', 'yellow', 'green', 'cyan'). It is measured as an angle around the central vertical axis of the color model, typically ranging from 0 to 360 degrees. By convention, Red is at 0°, Green is at 120°, and Blue is at 240°. The Hue channel of an image is a grayscale image where pixel intensity represents this angle.

    -   **Saturation (S):** This represents the "purity" or "vibrancy" of the color. It is a measure of the degree to which a pure color is diluted by white light. Saturation is measured as the radial distance from the central axis of the color model, typically ranging from 0 to 1 (or 0 to 100%).
        -   A saturation of 1 (or 100%) means the color is completely pure and undiluted (e.g., the most vivid red possible).
        -   A saturation of 0 means the color is completely desaturated, resulting in a shade of gray. For any pixel with S=0, the Hue is undefined and irrelevant.

    -   **Intensity (I):** Also known as Value (V) or Lightness (L) in similar models (HSV, HSL), this represents the brightness or luminance of the color. It is measured along the central vertical axis, ranging from 0 (black) to 1 (white). This component is completely decoupled from the color information (Hue and Saturation). The Intensity channel of a color image looks like a standard grayscale conversion of that image.

    ### Why is this separation so useful in Image Processing?

    By separating intensity from the chrominance (color) components, we can perform many processing tasks more effectively and intuitively:

    1.  **Contrast Enhancement:** As mentioned, applying histogram equalization to the 'I' channel of an HSI image will improve its contrast across the entire image without changing the original colors. The H and S channels remain untouched.

    2.  **Color Segmentation:** It is much easier to segment objects of a specific color in HSI space. For example, to find all "red" objects in an image, you don't need a complex rule like `R > threshold_r AND G < threshold_g AND B < threshold_b`. Instead, you can simply define a range for the Hue channel (e.g., pixels with Hue between 340° and 20°). You can further refine this by setting thresholds for Saturation (to avoid grayish reds) and Intensity (to avoid very dark or very bright spots). This is far more robust to changes in lighting conditions than RGB-based segmentation.

    3.  **Color Manipulation:** Artists and designers use HSI-like models to intuitively adjust colors. Want to make the image more vibrant? Increase the Saturation channel. Want to change the color of an object from red to blue? Select the object and shift its Hue values.

    4.  **Image Analysis:** The HSI model is valuable for analyzing scenes. For example, in computer vision for autonomous vehicles, the Hue channel can be very effective for identifying the color of traffic lights or road signs, regardless of whether it's a bright sunny day or a dim, overcast one (which mainly affects the Intensity channel).
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

        st.markdown("Notice how the **Intensity** channel looks like a standard grayscale version of the original image. The **Hue** channel shows where the dominant colors are (e.g., different shades of gray representing different angles on the color wheel), and the **Saturation** channel is brightest in areas of pure, vibrant color and darkest in grayish or muted areas.")

    st.success("**Key Takeaway:** The HSI color model is a perception-based model that separates color information (Hue, Saturation) from brightness (Intensity), making it ideal for tasks where you want to modify image brightness without affecting its colors.")


elif topic == "Laplacian of Gaussian (LoG)":
    st.header("4. Laplacian of Gaussian (LoG)")
    st.markdown("""
    The Laplacian of Gaussian (LoG) is a powerful and widely used edge detection operator. It is known as a "second-order" or "second-derivative" edge detector because it looks for **zero-crossings** in the second derivative of the image to find edges. This is in contrast to first-order detectors like the Sobel or Prewitt operators, which look for peaks (local maxima) in the first derivative (the gradient).

    ### The Two-Step Process: Taming the Laplacian

    The LoG operator elegantly combines two distinct operations into a single, more robust one. To understand its brilliance, we must first look at the components.

    1.  **The Laplacian Operator ($\nabla^2$):** The Laplacian is a second-derivative operator. In 2D, it is defined as $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$. It is excellent at highlighting regions of rapid intensity change (like edges) and produces a double edge (one on each side of the original edge). However, it has a major weakness: **it is extremely sensitive to noise.** Because differentiation amplifies noise, applying the Laplacian directly to a noisy image often results in an output where the noise completely overwhelms the actual edge information.

    2.  **The Gaussian Filter ($G$):** The Gaussian filter is a low-pass filter used for smoothing or blurring an image. Its primary purpose in this context is **noise reduction.** By pre-smoothing the image with a Gaussian filter, we can suppress the noise before it gets amplified by the Laplacian.

    The LoG method combines these two steps. Because both Gaussian filtering and the Laplacian are linear operations, they are commutative. This means that instead of first filtering the image with a Gaussian and then applying the Laplacian (`∇²[G * f]`), we can first convolve the Gaussian function with the Laplacian operator to create a single new kernel (`[∇²G]`) and then convolve that kernel with the image (`[∇²G] * f`). This is computationally more efficient.

    The final filtered image contains positive and negative values. The edges are located at the **zero-crossings** – the points where the pixel values transition from positive to negative or vice-versa.
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
    **Why is there a negative sign in the LoG formula? A Deeper Look.**

    The negative sign is included purely by **convention** to create a kernel that is more visually intuitive. Let's trace the logic:
    - The standard 2D Gaussian function has a positive peak at the center.
    - The Laplacian operator measures the "concavity" of a function. At the very peak of the Gaussian bell curve, the function is maximally concave (curving downwards), so its second derivative is **negative**. In the surrounding "skirt" of the bell curve, the curvature becomes convex, so the second derivative is **positive**.
    - Therefore, the raw Laplacian of a Gaussian, `∇²G`, naturally produces a kernel with a **negative value at its center** and a ring of positive values around it.
    - To make the kernel's visual representation more intuitive and easier to work with, we multiply the entire function by -1. This inverts the kernel, creating the familiar "Mexican Hat" shape with a **positive central peak and a negative surrounding ring**. This has no effect on the location of the zero-crossings, which is what we ultimately care about for edge detection.
    """)

    st.subheader("Interactive Demo")
    col1, col2 = st.columns(2)
    with col1:
        st.image(gray_img, caption="Original Grayscale Image", use_column_width=True)

    with col2:
        kernel_size = st.slider("Select Gaussian kernel size (odd)", 3, 21, 5, 2)
        sigma = st.slider("Sigma (σ) of Gaussian", 0.1, 5.0, 1.4, 0.1)

        st.markdown(f"""
        The `kernel_size` and `sigma` (σ) are related. A larger sigma creates more smoothing, reducing more noise but potentially blurring finer edges. The kernel size should be large enough to properly represent the Gaussian function defined by sigma (a common rule of thumb is `kernel_size ≈ 6σ`).
        """)

        # Step 1: Apply Gaussian blur to reduce noise
        blurred_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), sigma)

        # Step 2: Apply the Laplacian operator. We use a 64-bit float format (CV_64F)
        # to capture the positive and negative values around the zero-crossings.
        log_img = cv2.Laplacian(blurred_img, cv2.CV_64F, ksize=kernel_size)

        # For visualization purposes, we can take the absolute value. However, for true edge
        # detection, one would implement an algorithm to find the zero-crossing contours.
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
