#include <glib.h>
#include <cmath>
#include <string>
#include "json.hpp"
#include <iostream>
#include <fstream>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

// Function that return
// dot product of two vector array.
float dotProduct(float vect_A[], float vect_B[])
{
 
    float product = 0;
 
    // Loop for calculate cot product
    for (int i = 0; i < 128; i++)
    {
        product += vect_A[i] * vect_B[i];
    }

    return product;
}

extern "C" void
object_meta_data1(NvDsBatchMeta *batch_meta)
{
    static guint use_device_mem = 0;

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                continue;

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];

                if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                    cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i], 
                    info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }

            /* Parse output tensor and fill detection results into objectList. */
            std::vector < NvDsInferLayerInfo > outputLayersInfo (meta->output_layers_info, 
            meta->output_layers_info + meta->num_output_layers);
            std::vector < NvDsInferObjectDetectionInfo > objectList;
        }

    }

    use_device_mem = 1 - use_device_mem;
    return;
}


extern "C" void
object_meta_data2(NvDsBatchMeta *batch_meta)
{
    static guint use_device_mem = 0;

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        /* Iterate object metadata in frame */
        for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            /* Iterate user metadata in object to search SGIE's tensor data */
            for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                continue;

                /* convert to tensor metadata */
                NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
                g_printf("Num output layers  : %d \n", meta->num_output_layers);
                for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                    NvDsInferLayerInfo *info = &meta->output_layers_info[i];

                    info->buffer = meta->out_buf_ptrs_host[i];
                    float (*array)[128] = (float (*)[128]) info->buffer;
                    std::vector<float> embeddings_detection;

                    g_print("Shape  : %d \n", info->inferDims.numElements);
                    g_print("128d Tensor [ ");
                    for (unsigned int k = 0; k < info->inferDims.numElements; k++) {
                        g_print("%f, ", (*array)[k]);
                        embeddings_detection.insert(embeddings_detection.end(), (*array)[k]);
                    }
                    g_print("] \n");

                    // Config
                    g_print("[INFO] Loading config...\n");
                    std::ifstream is("config.json");
                    nlohmann::json config;
                    is >> config;

                    std::string embeddings_str = config["embeddings"];
                    is.open(embeddings_str);
                    g_print("[INFO] Embeddings: %s\n", embeddings_str.c_str());

                    is.close();
                    is.clear();

                    // np.dot(vectorA, vectorB.T) # 計算 cos 距離相似度
                    float testdata[128] = {-0.131958, 0.171631, -1.066406, -0.046051, -0.542480, 
                    1.295898, 0.031189, -0.773438, -1.206055, -0.581543, 
                    0.412598, 0.697754, -0.928223, 0.901367, 0.478516, 
                    0.695801, 0.492432, -2.054688, -0.072693, -1.447266, 
                    -0.823730, -1.295898, -0.913086, -0.835938, 1.700195, 
                    0.327881, 0.222290, 2.195312, 0.513672, 1.453125, 1.532227, 
                    0.705566, 0.378662, -0.474121, -0.612305, 0.812012, -1.908203, 
                    1.114258, 2.558594, 0.106567, 0.440430, 0.936523, -0.401367, 
                    -1.156250, 1.586914, 1.486328, -1.141602, -0.426758, -1.163086, 
                    0.638184, -0.403320, 1.660156, 1.959961, -0.277344, 0.269775, 
                    2.257812, -0.796387, -1.167969, 1.911133, -0.065674, 0.465332, 
                    0.897949, 0.673828, -0.695312, -0.184692, -0.710449, -0.640625, 
                    -1.031250, -0.834473, 0.130859, 0.666504, 0.241089, 0.082092, 
                    -1.273438, -0.141113, 0.814453, 0.174194, 1.113281, 0.942383, 
                    1.048828, 1.225586, 0.575684, -0.397705, -0.143799, 0.948730, 
                    -0.328857, 0.378418, 0.672852, -0.279297, -0.359863, 0.820312, 
                    0.305664, 0.212769, 0.607422, 0.833496, 1.150391, 0.120667, 
                    -0.444092, -1.832031, 0.910645, -0.773438, -0.423096, 0.691406, 
                    -0.431641, -0.282227, -0.312500, 0.298584, 1.341797, -0.256104, 
                    0.949219, 0.113525, 0.662109, -0.225830, -0.545410, 0.626465, 
                    0.143677, 0.003296, 2.332031, -0.203613, -1.691406, 1.791992, 
                    0.511230, 3.275391, 0.285645, 1.069336, -1.086914, 0.319824, 
                    -0.372314};
                    float printdot = dotProduct(embeddings_detection.data(), testdata);
                    g_print("Dot Product : %f \n", printdot);
                    if (printdot > 90) {
                        g_print("[INFO] Detected: %s\n", "Ming");
                    } else {
                        g_print("[INFO] Detected: %s\n", "not Ming");
                    }

                    if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                        cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i], 
                        info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                    }
                }

            }

        }

    }

    use_device_mem = 1 - use_device_mem;
    return;
}