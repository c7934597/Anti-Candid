#include <glib.h>
#include <cmath>
#include <string>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

// #define PGIE_NET_WIDTH 416
// #define PGIE_NET_HEIGHT 416

// unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
// unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;

extern "C" void
object_meta_data1(NvDsBatchMeta *batch_meta)
{
    static guint use_device_mem = 0;
    // static NvDsInferNetworkInfo networkInfo { PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3 };

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

        // #if NVDS_VERSION_MAJOR >= 5
        //     if (nvds_lib_major_version >= 5) {
        //         if (meta->network_info.width != networkInfo.width || 
        //         meta->network_info.height != networkInfo.height ||
        //         meta->network_info.channels != networkInfo.channels) {
        //             g_error ("failed to check pgie network info\n");
        //         }
        //     }
        // #endif
        }

        // /* Iterate obj metadata in frames to search PGIE's tensor metadata */
        // for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
        //     NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
        //     if (obj_meta->obj_label[0] != '\0')
        //     {
        //         printf("face \n");
        //     }
        // }
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
        // NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        // /* Iterate user metadata in frames to search PGIE's tensor metadata */
        // for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
        //     NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        //     if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        //         continue;

        //     /* convert to tensor metadata */
        //     NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        //     //std::cout<<"size "<< *(unsigned int *)meta->num_output_layers;
        //     for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        //         NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                
        //         info->buffer = meta->out_buf_ptrs_host[i];
        //         float (*array)[130] = (float (*)[130]) info->buffer;

        //         std::cout<<"Shape "<<info->inferDims.numElements<<std::endl;
        //         std::cout<<"128d Tensor"<<std::endl;
        //         for(int m =0;m<info->inferDims.numElements;m++){
        //             std::cout<<" "<< (*array)[m];
        //         }
        //         //double* ptr = (double*)meta->out_buf_ptrs_host[0];  // output layer 0
        //         //for( size_t i=0; i<info->inferDims.numElements; i++ )
        //         //{
        //             //std::cout << "Tensor " << i << ": " << ptr[i] << std::endl;
        //             //std::cout  << i << ": " << ptr[i];
        //         // }
                
        //         if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
        //             cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
        //                 info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        //         }

        //     }

        // }

        // /* Iterate obj metadata in frames to search PGIE's tensor metadata */
        // for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
        //     NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
        //     if (obj_meta->obj_label[0] != '\0')
        //     {
        //         printf("face \n");
        //     }
        // }
    }

    use_device_mem = 1 - use_device_mem;
    return;
}