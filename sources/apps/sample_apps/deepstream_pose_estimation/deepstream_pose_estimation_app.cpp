// Copyright 2020 - NVIDIA Corporation
// SPDX-License-Identifier: MIT

#include "post_process.cpp"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "deepstream_common.h"
#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>

#define EPS 1e-6

#define MAX_DISPLAY_LEN 64

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 960

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

gint frame_number = 0;
gint countWarning = 0;

/*Method to parse information returned from the model*/
std::tuple<Vec2D<int>, Vec3D<float>>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
  Vec1D<int> counts;
  Vec3D<int> peaks;

  float threshold = 0.1;
  int window_size = 5;
  int max_num_parts = 2;
  int num_integral_samples = 7;
  float link_threshold = 0.1;
  int max_num_objects = 100;

  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  void *paf_data = tensor_meta->out_buf_ptrs_host[1];
  NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;

  /* Finding peaks within a given window */
  find_peaks(counts, peaks, cmap_data, cmap_dims, threshold, window_size, max_num_parts);
  /* Non-Maximum Suppression */
  Vec3D<float> refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size);
  /* Create a Bipartite graph to assign detected body-parts to a unique person in the frame */
  Vec3D<float> score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples);
  /* Assign weights to all edges in the bipartite graph generated */
  Vec3D<int> connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts);
  /* Connecting all the Body Parts and Forming a Human Skeleton */
  Vec2D<int> objects = connect_parts(connections, topology, counts, max_num_objects);
  return {objects, refined_peaks};
}

float convert_radian_to_degrees(float radian){ 
    float pi = 3.14159; 
    return (radian * (180/pi)); 
}

float get_angle(float a0, float a1, float b0, float b1)
{
    float del_y = a1-b1;
    float del_x = b0-a0;
    if (del_x == 0)
      del_x = 0.1;
    float angle = 0;

    if (del_x > 0 && del_y > 0)
      angle = convert_radian_to_degrees(atan(del_y / del_x));
    if (del_x < 0 && del_y > 0)
      angle = convert_radian_to_degrees(atan(del_y / del_x)) + 180;

    return angle;
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
  int K = topology.size();
  int count = objects.size();
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  bool IsWarning = false;
  for (auto &object : objects)
  {
    int C = object.size();
    for (int j = 0; j < C; j++)
    {
      int k = object[j];
      if (k >= 0)
      {
        auto &peak = normalized_peaks[j][k];
        int x = peak[1] * MUXER_OUTPUT_WIDTH;
        int y = peak[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
        cparams.xc = x;
        cparams.yc = y;
        cparams.radius = 8;
        cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
        cparams.has_bg_color = 1;
        cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_circles++;
      }
    }

    for (int k = 0; k < K; k++)
    {
      int c_a = topology[k][2];
      int c_b = topology[k][3];
      if (object[c_a] >= 0 && object[c_b] >= 0)
      {
        auto &peak0 = normalized_peaks[c_a][object[c_a]];
        auto &peak1 = normalized_peaks[c_b][object[c_b]];
        if (k == 7 || k ==8)
        {
          float angle = get_angle(peak0[1], peak0[0], peak1[1], peak1[0]);
          if (angle > 0)
            IsWarning = true;
        }
        int x0 = peak0[1] * MUXER_OUTPUT_WIDTH;
        int y0 = peak0[0] * MUXER_OUTPUT_HEIGHT;
        int x1 = peak1[1] * MUXER_OUTPUT_WIDTH;
        int y1 = peak1[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
        lparams.x1 = x0;
        lparams.x2 = x1;
        lparams.y1 = y0;
        lparams.y2 = y1;
        lparams.line_width = 3;
        lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_lines++;
      }
    }
  }

  if(IsWarning)
  {
    countWarning=countWarning+1;
    printf("Warning %d\n", countWarning);
  }
  else
  {
    countWarning=0;
    printf("Safe\n");
  }

  if(countWarning == 30)
  {
    printf("============================Command===========================\n");
    // system("python3 dolock.py 192.168.110.44");
  }
}

/* pgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
         l_user = l_user->next)
    {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
      {
        NvDsInferTensorMeta *tensor_meta =
            (NvDsInferTensorMeta *)user_meta->user_meta_data;
        Vec2D<int> objects;
        Vec3D<float> normalized_peaks;
        tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
        create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
      }
    }

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
          Vec2D<int> objects;
          Vec3D<float> normalized_peaks;
          tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
          create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number =  %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    txt_params->font_params.font_name = "Mono";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

gboolean
link_element_to_tee_src_pad(GstElement *tee, GstElement *sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(tee), "src_%u");
  tee_src_pad = gst_element_request_pad(tee, padtemplate, NULL, NULL);

  if (!tee_src_pad)
  {
    g_printerr("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad(sinkelem, "sink");
  if (!sinkpad)
  {
    g_printerr("Failed to get sink pad from '%s'",
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }

  if (gst_pad_link(tee_src_pad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link '%s' and '%s'", GST_ELEMENT_NAME(tee),
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }
  ret = TRUE;

done:
  if (tee_src_pad)
  {
    gst_object_unref(tee_src_pad);
  }
  if (sinkpad)
  {
    gst_object_unref(sinkpad);
  }
  return ret;
}

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstCaps *caps1 = NULL, *caps2 = NULL;
  GstElement *pipeline = NULL, *source = NULL, *caps_v4l2src = NULL, *caps_vidconvsrc = NULL, *vidconvsrc = NULL, 
             *nvvidconvsrc = NULL,  * nvvidconv = NULL, *streammux = NULL, *pgie = NULL, *nvosd = NULL, *sink = NULL;

/* Add a transform element for Jetson*/
  GstElement *transform = NULL;

  GstCapsFeatures *feature = NULL;
  GstPad *sinkpad, *srcpad;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

  source = gst_element_factory_make ("v4l2src", "usb-cam-source");
  if (!source) {
    NVGSTDS_ERR_MSG_V ("Could not create 'usb-cam-source''");
    goto done;
  }

  caps_v4l2src = gst_element_factory_make("capsfilter", "v4l2src_caps");
  if (!caps_v4l2src) {
    NVGSTDS_ERR_MSG_V ("Could not create 'v4l2src_caps'");
    goto done;
  }
  caps1 = gst_caps_new_simple ("video/x-raw",
          "width", G_TYPE_INT, 640, "height", G_TYPE_INT,
         480, "framerate", GST_TYPE_FRACTION,
          30, 1, NULL);

  caps_vidconvsrc = gst_element_factory_make("capsfilter", "nvmm_caps");
  if (!caps_vidconvsrc) {
    NVGSTDS_ERR_MSG_V ("Could not create 'nvmm_caps'");
    goto done;
  }
  caps2 = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
          "width", G_TYPE_INT, 640, "height", G_TYPE_INT,
         480, "framerate", GST_TYPE_FRACTION,
          30, 1, NULL);

  /* videoconvert to make sure a superset of raw formats are supported */
  vidconvsrc = gst_element_factory_make("videoconvert", "convertor_src1");
  if (!vidconvsrc) {
    NVGSTDS_ERR_MSG_V ("Could not create 'convertor_src1'");
    goto done;
  }

  /* nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API) */
  nvvidconvsrc = gst_element_factory_make("nvvideoconvert", "convertor_src2");
  if (!nvvidconvsrc) {
    NVGSTDS_ERR_MSG_V ("Could not create 'convertor_src2'");
    goto done;
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
  if (!nvvidconv) {
    NVGSTDS_ERR_MSG_V ("Could not create 'convertor'");
    goto done;
  }

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  if (!pgie) {
    NVGSTDS_ERR_MSG_V ("Could not create 'primary-nvinference-engine'");
    goto done;
  }

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    NVGSTDS_ERR_MSG_V ("Could not create 'nv-onscreendisplay'");
    goto done;
  }

  /* Finally render the osd output */
  transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
  if (!transform)
  {
    g_printerr("One tegra element could not be created. Exiting.\n");
    return -1;
  }

  sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
  if (!sink) {
    NVGSTDS_ERR_MSG_V ("Could not create 'nvvideo-renderer'");
    goto done;
  }

  g_object_set (G_OBJECT (nvvidconvsrc), "gpu-id", 0,
    "nvbuf-memory-type", 0, NULL);

  g_object_set (G_OBJECT (caps_v4l2src), "caps", caps1, NULL);

  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps2, 0, feature);
  g_object_set (G_OBJECT (caps_vidconvsrc), "caps", caps2, NULL);

  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", 1,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set(G_OBJECT(pgie), "output-tensor-meta", TRUE,
               "config-file-path", "deepstream_pose_estimation_config.txt", NULL);

  g_object_set(G_OBJECT(sink), "sync", FALSE, NULL);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many(GST_BIN(pipeline), 
                   source, caps_v4l2src, vidconvsrc, nvvidconvsrc, caps_vidconvsrc, 
                   streammux, pgie, nvvidconv, nvosd, sink, transform, NULL);

  if (!gst_element_link_many (source, caps_v4l2src,
          vidconvsrc, nvvidconvsrc, caps_vidconvsrc, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

  sinkpad = gst_element_get_request_pad(streammux, "sink_0");
  if (!sinkpad)
  {
    g_printerr("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad(caps_vidconvsrc, "src");
  if (!srcpad)
  {
    g_printerr("Caps_vidconvsrc request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link caps_vidconvsrc to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref(sinkpad);
  gst_object_unref(srcpad);

  if (!gst_element_link_many (streammux, pgie,
          nvvidconv, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  GstPad *pgie_src_pad;
  pgie_src_pad = gst_element_get_static_pad(pgie, "src");
  if (!pgie_src_pad)
    g_print("Unable to get pgie src pad\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_pad_buffer_probe, (gpointer)sink, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, (gpointer)sink, NULL);

  /* Set the pipeline to "playing" state */
  g_print("Now playing\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

done:
  if (caps1)
    gst_caps_unref (caps1);
    
  if (caps2)
    gst_caps_unref (caps2);

  return 0;
}