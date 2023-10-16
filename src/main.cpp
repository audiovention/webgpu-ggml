#include <chrono>
#include <iostream>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"

#include <webgpu/webgpu.h>

void print_tensor(struct ggml_tensor * tensor, const char* name) {
    printf("%s:\n", name);
    for (int i=0; i<tensor->ne[3]; i++) {
        for (int j=0; j<tensor->ne[2]; j++) {
            for (int k=0; k<tensor->ne[1]; k++) {
                for (int l=0; l<tensor->ne[0]; l++) {
                    if (tensor->ne[3] > 1) {
                        printf("y[%d][%d][%d][%d] = %5.10f\n", l, k, j, i, ggml_get_f32_nd(tensor, l, k, j, i));
                    } else {
                        if (tensor->ne[2] > 1) {
                            printf("y[%d][%d][%d] = %5.10f\n", l, k, j, ggml_get_f32_nd(tensor, l, k, j, i));
                        } else {
                            if (tensor->ne[1] > 1) {
                                printf("y[%d][%d] = %5.10f\n", l, k, ggml_get_f32_nd(tensor, l, k, j, i));
                            } else {
                                printf("y[%d] = %5.10f\n", l, ggml_get_f32_nd(tensor, l, k, j, i));
                            }
                        }
                    }
                }
            }
        }
    }
    
}

int main()
{


	// 1. We create a descriptor
	WGPUInstanceDescriptor desc = {};
	desc.nextInChain = nullptr;

	// 2. We create the instance using this descriptor
	WGPUInstance instance = wgpuCreateInstance(&desc);

	// 3. We can check whether there is actually an instance created
	if (!instance) {
		std::cerr << "Could not initialize WebGPU!" << std::endl;
		return 1;
	}

	// 4. Display the object (WGPUInstance is a simple pointer, it may be
	// copied around without worrying about its size).
	std::cout << "WGPU instance: " << instance << std::endl;







    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
    };


    // memory allocation happens here
    struct ggml_context* ctx = ggml_init(params);

    // ggml_backend_t backend = NULL;
    // backend = ggml_backend_cpu_init();

    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 9);


    ggml_set_f32_1d(x, 0, 0.0001f);
    ggml_set_f32_1d(x, 1, 0.001f);
    ggml_set_f32_1d(x, 2, 0.01f);
    ggml_set_f32_1d(x, 3, 0.1f);
    ggml_set_f32_1d(x, 4, 1.0f);
    ggml_set_f32_1d(x, 5, 10.0f);
    ggml_set_f32_1d(x, 6, 100.0f);
    ggml_set_f32_1d(x, 7, 1000.0f);
    ggml_set_f32_1d(x, 8, 10000.0f);



    struct ggml_tensor* y = ggml_tanh(ctx, x);


    struct ggml_cgraph gf = ggml_build_forward(y);


    ggml_graph_compute_with_ctx(ctx, &gf, 4); 

    print_tensor(y, "y");






	// 5. We clean up the WebGPU instance
	wgpuInstanceRelease(instance);


    return 0;
}