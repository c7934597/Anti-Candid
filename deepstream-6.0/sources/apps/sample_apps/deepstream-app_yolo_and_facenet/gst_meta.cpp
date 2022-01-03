#include <glib.h>
#include <cmath>
#include <string>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"


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
                printf("Num output layers  : %d \n", meta->num_output_layers);
                for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                    NvDsInferLayerInfo *info = &meta->output_layers_info[i];

                    info->buffer = meta->out_buf_ptrs_host[i];
                    float (*array)[130] = (float (*)[130]) info->buffer;

                    printf("Shape  : %d \n", info->inferDims.numElements);
                    printf("128d Tensor [ ");
                    for (unsigned int k = 0; k < info->inferDims.numElements; k++) {
                        printf("%f ", (*array)[k]);
                    }
                    printf("] \n");

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