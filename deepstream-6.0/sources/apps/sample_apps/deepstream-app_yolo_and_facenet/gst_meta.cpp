#include "gstnvdsmeta.h"

#include <cmath>
#include <string>

extern "C" void
object_meta_data1(NvDsBatchMeta *batch_meta)
{
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
   
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (obj_meta->obj_label[0] != '\0')
            {
                printf("face \n");
            }
        }
    }
    return;
}