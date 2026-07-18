# Manual Prompt Guide for CS898BA Homework One

This guide provides estimated prompts that a user could use to manually complete the CS898BA Homework One assignment, broken down by the assignment's sections.

## Part 1: Project Setup

### 1.1 Repository Creation and Initialization
**User Prompt:** "Create a new GitHub repository named `AbdulAleemMohammed-CS898BA-Project1`. Initialize it with a README.md, and add a Python file named `hello_world.py` containing `print("Hello World!")`. Also, create an empty file named `AI_Log.md`."

### 1.2 Initial Commit
**User Prompt:** "Commit the initial files (`README.md`, `hello_world.py`, `AI_Log.md`) to the repository with the commit message 'Initial commit: Hello World and AI Log'."

### 1.3 AI Usage Logging
**User Prompt:** "For every significant AI interaction during this assignment, add an entry to `AI_Log.md` following this format: `| Date and Time | Prompt | Tool | Response Synopsis | Change |`"

## Part 2: Basic Image Analysis

### 2.1 Image Statistics
**User Prompt:** "Using Python and OpenCV, load the `alien_image.png`. Calculate and print the minimum, maximum, average, median, mode, skew, range, standard deviation, and variance for each of its color channels (Red, Green, Blue)."

### 2.2-2.4 Image Conversions and Normalization
**User Prompt:** "Convert the `alien_image.png` to greyscale, binary, HSV, CIELAB, and HLS color spaces. Save each converted image to a new file. Then, take the HSV image, perform histogram equalization on its V (Value) channel to normalize lighting, and convert this normalized HSV image back to RGB. Save this final image."

### 2.5-2.6 Affine Transformations
**User Prompt:** "Take the 7 images generated so far (original, greyscale, binary, HSV, CIELAB, HLS, and the normalized RGB image). For each of these 7 images, apply two unique affine transformations (e.g., translation, rotation, scaling, shear, or combinations). Ensure no two transformations are exactly the same across all 14 new images. Save each of these 14 transformed images to new files."

### 2.7-2.8 Gaussian Blur
**User Prompt:** "Now, take all 21 images (the initial 7 and the 14 affine-transformed ones). Apply a Gaussian blur to each of these 21 images using the following sigma values: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. Save each blurred image to a new file. Discuss how the level of sigma affects the image's appearance and detail retention."

## Part 3: Edge Detection

### 3.1-3.2 Image Subset Selection
**User Prompt:** "From the total of 168 images (original, converted, affine, and blurred), randomly select 42 images to form a subset for edge detection analysis."

### 3.3-3.4 Edge Detection Application
**User Prompt:** "For each image in the selected subset of 42, apply the following edge detection techniques: Sobel, Laplacian, Canny, and Prewitt. Save the resulting edge-detected images."

### 3.5 Edge Detection Discussion
**User Prompt:** "Analyze and discuss the pros and cons of each edge detection technique (Sobel, Laplacian, Canny, Prewitt) based on their performance on the image subset. Recommend which technique works best for identifying the 'alien' figure in this specific image set, providing reasoning."

### 3.6-3.8 Result Visualization
**User Prompt:** "For each of the 42 images in the subset, create a 5-panel plot showing the original image and its corresponding Sobel, Laplacian, Canny, and Prewitt edge-detected versions. Save these plots as image files. Then, select 6 random plots and embed them into the `README.md` file, along with a brief description of the processing pipeline for each displayed plot."

## Part 4: Submission

### 4.1 Final Repository Preparation
**User Prompt:** "Ensure the `README.md` is comprehensive, detailing the project, setup, execution, results, and discussions. Confirm that `AI_Log.md` is up-to-date with all AI interactions. Zip the entire project directory for submission."

### 4.2 GitHub Submission
**User Prompt:** "Upload the zipped project to a new GitHub repository and provide the public link for submission."
