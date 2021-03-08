#include "post_process.cpp"

/*#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"*/

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
#define MUXER_OUTPUT_WIDTH 416
#define MUXER_OUTPUT_HEIGHT 416

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33333

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

gint frame_number = 0;

#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define CLIENT_PORT 21567 //目標地址埠號
#define CLIENT_IP "192.168.110.44" //目標地址IP

gint PoseWarning = 0;
gint PeopleWarning = 0;
gint SuspiciousItemWarning = 0;
gint PoseWarningLimit = 15;
gint PeopleWarningLimit = 30;
gint SuspiciousItemWarningLimit = 15;
gchar lockbuf[]="LOCK";

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

extern "C" void
send_lock_socket(char buf[], bool detection){
  int sockfd = 0;
  int so_broadcast = 1;
  struct sockaddr_in server_addr, client_addr;

  /*Create an IPv4 UDP socket*/
  if((sockfd = socket(AF_INET, SOCK_DGRAM, 0))<0){
    perror("socket");
    return;
  }

  /*SO_BROADCAST: broadcast attribute*/
  if(setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &so_broadcast, sizeof(so_broadcast))<0){
    perror("setsockopt");
    return;
  }

  server_addr.sin_family = AF_INET; /*IPv4*/
  server_addr.sin_port = htons(INADDR_ANY); /*All the port*/
  server_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST); /*Broadcast address*/

  if((bind(sockfd, (struct sockaddr*)&server_addr, sizeof(struct sockaddr))) != 0){
    perror("bind");
    return;
  }

  client_addr.sin_family = AF_INET; /*IPv4*/
  client_addr.sin_port = htons(CLIENT_PORT);  /*Set port number*/
  client_addr.sin_addr.s_addr = inet_addr(CLIENT_IP); /*Set the broadcast address*/
  int clientlen = sizeof(client_addr);

  /*Use sendto() to send messages to client*/
  /*sendto() doesn't need to be connected*/
  if((sendto(sockfd, buf, strlen(buf), 0, (struct sockaddr*)&client_addr, (socklen_t)clientlen)) < 0){
    perror("sendto");
    return;
  }
  else{
    printf("send msg %s\n", buf);
    if(detection){
      PoseWarning = 0;
      PeopleWarning = 0;
      SuspiciousItemWarning = 0;
    }
  }

  close(sockfd);  /*close socket*/
  return;
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
  int K = topology.size();
  int countPeople = objects.size();
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
        cparams.radius = 6;
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
        //g_print("%d\n",k);
        lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1}; 
        dmeta->num_lines++;
      }
    }
  }

  if(IsWarning)
  {
    PoseWarning++;
    printf("Pose Warning : %d \n", PoseWarning);
  }
  else
    PoseWarning=0;

  if(countPeople>1)
  {
    PeopleWarning++;
    printf("Over People Warning : %d \n", PeopleWarning);
  }
  else
    PeopleWarning=0;

  if(PoseWarning == PoseWarningLimit || PeopleWarning == PeopleWarningLimit)
  {
    send_lock_socket(lockbuf , true);
  }
}

extern "C" void
pose_meta_data(NvDsBatchMeta *batch_meta)
{
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_user = NULL;
   
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
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
    return;
}

extern "C" void
object_meta_data(NvDsBatchMeta *batch_meta)
{
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
   
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        bool IsWarning = false;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (obj_meta->obj_label[0] != '\0')
            {
              IsWarning = true;
            }
        }
        if(IsWarning)
        {
          SuspiciousItemWarning++;
          printf("Suspicious Item Warning : %d \n", SuspiciousItemWarning);
        }
        else
          SuspiciousItemWarning = 0;
    }

    if(SuspiciousItemWarning == SuspiciousItemWarningLimit)
    {
      send_lock_socket(lockbuf , true);
    }
    return;
}