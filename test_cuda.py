import tensorflow as tf

# build_info = tf.sysconfig.get_build_info()
# print("Build info ", build_info)

sess = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print("GPU available? ", sess)
built = tf.test.is_built_with_cuda()
print("tf is built with CUDA? ", built)
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print("Num GPUs used: ", len(gpus))
print("Num CPUs used: ", len(cpus))