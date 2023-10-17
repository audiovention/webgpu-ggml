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
                        printf("%s[%d][%d][%d][%d] = %5.10f\n", name, l, k, j, i, ggml_get_f32_nd(tensor, l, k, j, i));
                    } else {
                        if (tensor->ne[2] > 1) {
                            printf("%s[%d][%d][%d] = %5.10f\n", name, l, k, j, ggml_get_f32_nd(tensor, l, k, j, i));
                        } else {
                            if (tensor->ne[1] > 1) {
                                printf("%s[%d][%d] = %5.10f\n", name, l, k, ggml_get_f32_nd(tensor, l, k, j, i));
                            } else {
                                printf("%s[%d] = %5.10f\n", name, l, ggml_get_f32_nd(tensor, l, k, j, i));
                            }
                        }
                    }
                }
            }
        }
    }
    
}


static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}


#if MY_GGML_USE_METAL
#include "ggml-metal.h"
void runOnMetal()
{
    std::cout <<"Running on Metal" << std::endl;
    ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);

//    ggml_backend_t backend = NULL;
//    backend = ggml_backend_metal_init();
//    if (backend) {
//        ggml_backend_free(backend);
//    }


    struct ggml_init_params params = {
            .mem_size = 16 * 1024 * 1024,
            .mem_buffer = NULL,
        };

    struct ggml_context* ctx = ggml_init(params);
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    ggml_set_f32_1d(x, 0, 0.01f);
    ggml_set_f32_1d(x, 1, 0.1f);
    ggml_set_f32_1d(x, 2, 1.0f);
    ggml_set_f32_1d(x, 3, 10.0f);
    ggml_set_f32_1d(x, 4, 100.0f);
    struct ggml_tensor* y = ggml_silu(ctx, x);
    struct ggml_cgraph gf = ggml_build_forward(y);


    auto ctxm = ggml_metal_init(1);
    ggml_metal_add_buffer(ctxm, "base", ggml_get_mem_buffer(ctx), ggml_get_mem_size(ctx), 0);

    ggml_metal_set_tensor(ctxm, x);
    ggml_metal_graph_compute(ctxm, &gf);
    ggml_metal_get_tensor(ctxm, y);

    print_tensor(y, "ym");



    if (ctxm) ggml_metal_free(ctxm);
    if (ctx) ggml_free(ctx);

}
#endif


int main()
{
#if MY_GGML_USE_METAL
    runOnMetal();
#endif

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

    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    ggml_set_f32_1d(x, 0, 0.01f);
    ggml_set_f32_1d(x, 1, 0.1f);
    ggml_set_f32_1d(x, 2, 1.0f);
    ggml_set_f32_1d(x, 3, 10.0f);
    ggml_set_f32_1d(x, 4, 100.0f);



    struct ggml_tensor* y = ggml_silu(ctx, x);


    struct ggml_cgraph gf = ggml_build_forward(y);


    ggml_graph_compute_with_ctx(ctx, &gf, 4); 

    print_tensor(y, "y");






	// 5. We clean up the WebGPU instance
	wgpuInstanceRelease(instance);


    return 0;
}