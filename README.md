You need to install **CUDA** and **cuDNN** for your NVIDIA RTX 3050 on Windows 11. Follow these steps:

### **Step 1: Install NVIDIA GPU Driver**
1. Download the latest **NVIDIA Game Ready or Studio Driver** from:  
   🔗 [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx)  
2. Install it and restart your system.

### **Step 2: Install CUDA Toolkit**
1. Go to the **CUDA Toolkit Download** page:  
   🔗 [CUDA Download](https://developer.nvidia.com/cuda-downloads)  
2. Select:
   - **Operating System**: Windows  
   - **Architecture**: x86_64  
   - **Version**: Choose the latest **CUDA 12.x** (not CUDA 10, as your error suggests an older version).  
   - **Installer Type**: Local or Network  
3. Download and install it.

### **Step 3: Install cuDNN**
1. Download **cuDNN** from:  
   🔗 [cuDNN Download](https://developer.nvidia.com/cudnn) (You need an NVIDIA account).  
2. Choose the **cuDNN version compatible with your CUDA version** (check TensorFlow's compatibility [here](https://www.tensorflow.org/install/source#gpu)).  
3. Extract the files and **copy**:
   - `bin` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - `include` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include`
   - `lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib`

### **Step 4: Set Environment Variables**
1. Open **System Properties** → "Advanced" → "Environment Variables."
2. Add these to **System Variables**:
   - **CUDA_PATH** = `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
   - Add to **Path**:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp
     ```

### **Step 5: Verify Installation**
1. Open **CMD** and run:  
   ```
   nvcc --version
   ```
   It should display the installed CUDA version.
2. Check if TensorFlow detects GPU:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

If you still get errors, ensure that:
- **TensorFlow version supports your CUDA/cuDNN version**.
- **You installed the latest NVIDIA driver**.

Let me know if you need help troubleshooting! 🚀
