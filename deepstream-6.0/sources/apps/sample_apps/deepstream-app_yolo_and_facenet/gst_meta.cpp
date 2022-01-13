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

#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

const std::vector<std::string> split(const std::string& str, const std::string& pattern) {
    std::vector<std::string> result;
    std::string::size_type begin, end;

    end = str.find(pattern);
    begin = 0;

    while (end != std::string::npos) {
        if (end - begin != 0) {
            result.push_back(str.substr(begin, end-begin)); 
        }    
        begin = end + pattern.size();
        end = str.find(pattern, begin);
    }

    if (begin != str.length()) {
        result.push_back(str.substr(begin));
    }
    return result;        
}

extern "C" void
object_meta_data2(NvDsBatchMeta *batch_meta)
{
    static guint use_device_mem = 0;

    // Config
    // g_print("[INFO] Loading config...\n");
    std::ifstream is("config.json");
    nlohmann::json config;
    is >> config;

    std::string embeddings_str = config["embeddings"];
    is.open(embeddings_str);
    // g_print("[INFO] Embeddings: %s\n", embeddings_str.c_str());

    is.close();
    is.clear();

    std::string str = embeddings_str.c_str();
    std::string pattern = ",";
    std::vector<std::string> ret = split(str, pattern);
    std::vector<float> embeddings_base;
    for(auto& s : ret){
        embeddings_base.insert(embeddings_base.end(), std::stof(s));
    }

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        /* Iterate object metadata in frame */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            if (obj_meta->obj_label[0] != '\0')
            {
                if(!strcmp(obj_meta->obj_label,"face"))
                {
                    /* Iterate user metadata in object to search SGIE's tensor data */
                    for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                        l_user = l_user->next) {
                        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
                            continue;
                        }

                        /* convert to tensor metadata */
                        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
                        // g_print("Num output layers  : %d \n", meta->num_output_layers);
                        for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                            NvDsInferLayerInfo *info = &meta->output_layers_info[i];

                            info->buffer = meta->out_buf_ptrs_host[i];
                            float (*array)[128] = (float (*)[128]) info->buffer;
                            std::vector<float> embeddings_detection;

                            // g_print("Shape  : %d \n", info->inferDims.numElements);
                            // g_print("128d Tensor [ ");
                            for (unsigned int k = 0; k < info->inferDims.numElements; k++) {
                                // g_print("%f, ", (*array)[k]);
                                embeddings_detection.insert(embeddings_detection.end(), (*array)[k]);
                            }
                            // g_print("] \n");

                            float printdot = dotProduct(embeddings_detection.data(), embeddings_base.data());
                            g_print("Dot Product : %f \n", printdot);
                            if (printdot > 60) {
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
                // else
                // {
                //     g_print("[INFO] Detected: %s\n", obj_meta->obj_label);
                // }
            }

        }

    }

    use_device_mem = 1 - use_device_mem;
    return;
}