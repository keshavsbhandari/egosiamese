import torchvision.transforms as T
import torch
import socket
import os

N_DEVICE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if socket.gethostname() == 'rainbow-panda' else 8
SERVER = "PANDAS" if socket.gethostname() == 'rainbow-panda' else "TRENTO"

M_SIAMESE_PATH = "cache/PANDAS_DEPTH_10_MOTION_SIAMESE.ckpt"
S_SIAMESE_PATH = "cache/DEPTH_10_SPATIAL_SIAMESE.ckpt"
MC_SIAMESE_PATH = "cache/MOTION_SIAMESE_CLASSIFIER.ckpt"
SC_SIAMESE_PATH = "cache/SPATIAL_SIAMESE_CLASSIFIER.ckpt"

MC_PATH = "cache/MOTION_CLASSIFIER.ckpt"
SC_PATH = "cache/SPATIAL_CLASSIFIER.ckpt"

SINGLE_VIDEO_POLICY = False

TRAIN_FROM_PICKLE = True
# minimum number of videos per subclass, the minimum number is 4
BALANCE_VIDEOS = 10

FEATURE_AGGREGATOR = True

ENC_SEQ_DIM = 128

DIMENSION = 768
IMG_W = 640
IMG_H = 320

PATCH_W = 192
PATCH_H = 128

DEPTH = 10
EMB_DIM = 768

TANGENT_PATCH = 64

S_LABELS = {'desk_work': 0,
            'driving': 1,
            'lunch': 2,
            'office_talk': 3,
            'ping-pong': 4,
            'playing_cards': 5,
            'playing_pool': 6,
            'running': 7,
            'sitting': 8,
            'stairs': 9,
            'standing': 10,
            'walking': 11}

S_LABELS_NAME = ['desk_work',
                 'driving',
                 'lunch',
                 'office_talk',
                 'ping-pong',
                 'playing_cards',
                 'playing_pool',
                 'running',
                 'sitting',
                 'stairs',
                 'standing',
                 'walking',]

M_LABELS = {'accelerate': 0,
            'at_computer': 1,
            'bounce_ball': 2,
            'breezeway': 3,
            'chalk_up': 4,
            'check_phone': 5,
            'crossing_street': 6,
            'decelerate': 7,
            'desk_work': 8,
            'doorway': 9,
            'down_stairs': 10,
            'drinking': 11,
            'driving': 12,
            'eating': 13,
            'follow_obj': 14,
            'hallway': 15,
            'hit': 16,
            'leaning': 17,
            'looking_at': 18,
            'napping': 19,
            'office_talk': 20,
            'ordering': 21,
            'pickup_ball': 22,
            'ping-pong': 23,
            'playing_cards': 24,
            'playing_pool': 25,
            'put_card': 26,
            'reach': 27,
            'running': 28,
            'serve': 29,
            'shooting': 30,
            'shuffle': 31,
            'sit_down': 32,
            'sitting': 33,
            'stand_up': 34,
            'standing': 35,
            'still': 36,
            'stop': 37,
            'take_card': 38,
            'turn_around': 39,
            'turn_left': 40,
            'turn_right': 41,
            'up_stairs': 42,
            'walking': 43,
            'writing': 44}

M_LABELS_NAME  = ['accelerate',
                  'at_computer',
                  'bounce_ball',
                  'breezeway',
                  'chalk_up',
                  'check_phone',
                  'crossing_street',
                  'decelerate',
                  'desk_work',
                  'doorway',
                  'down_stairs',
                  'drinking',
                  'driving',
                  'eating',
                  'follow_obj',
                  'hallway',
                  'hit',
                  'leaning',
                  'looking_at',
                  'napping',
                  'office_talk',
                  'ordering',
                  'pickup_ball',
                  'ping-pong',
                  'playing_cards',
                  'playing_pool',
                  'put_card',
                  'reach',
                  'running',
                  'serve',
                  'shooting',
                  'shuffle',
                  'sit_down',
                  'sitting',
                  'stand_up',
                  'standing',
                  'still',
                  'stop',
                  'take_card',
                  'turn_around',
                  'turn_left',
                  'turn_right',
                  'up_stairs',
                  'walking',
                  'writing']

LABELS = {'desk_work_desk_work': 0,
          'desk_work_napping': 1,
          'desk_work_sit_down': 2,
          'desk_work_stand_up': 3,
          'desk_work_turn_left': 4,
          'desk_work_turn_right': 5,
          'desk_work_writing': 6,
          'driving_accelerate': 7,
          'driving_decelerate': 8,
          'driving_driving': 9,
          'driving_still': 10,
          'driving_stop': 11,
          'driving_turn_left': 12,
          'driving_turn_right': 13,
          'lunch_drinking': 14,
          'lunch_eating': 15,
          'lunch_ordering': 16,
          'lunch_turn_left': 17,
          'lunch_turn_right': 18,
          'office_talk_check_phone': 19,
          'office_talk_office_talk': 20,
          'office_talk_reach': 21,
          'office_talk_turn_left': 22,
          'office_talk_turn_right': 23,
          'ping-pong_bounce_ball': 24,
          'ping-pong_hit': 25,
          'ping-pong_pickup_ball': 26,
          'ping-pong_ping-pong': 27,
          'ping-pong_serve': 28,
          'playing_cards_playing_cards': 29,
          'playing_cards_put_card': 30,
          'playing_cards_shuffle': 31,
          'playing_cards_take_card': 32,
          'playing_pool_chalk_up': 33,
          'playing_pool_check_phone': 34,
          'playing_pool_playing_pool': 35,
          'playing_pool_reach': 36,
          'playing_pool_shooting': 37,
          'playing_pool_turn_left': 38,
          'playing_pool_turn_right': 39,
          'running_looking_at': 40,
          'running_running': 41,
          'running_turn_around': 42,
          'sitting_at_computer': 43,
          'sitting_check_phone': 44,
          'sitting_follow_obj': 45,
          'sitting_reach': 46,
          'sitting_sitting': 47,
          'sitting_turn_left': 48,
          'sitting_turn_right': 49,
          'stairs_doorway': 50,
          'stairs_down_stairs': 51,
          'stairs_reach': 52,
          'stairs_turn_left': 53,
          'stairs_turn_right': 54,
          'stairs_up_stairs': 55,
          'standing_leaning': 56,
          'standing_standing': 57,
          'walking_breezeway': 58,
          'walking_crossing_street': 59,
          'walking_doorway': 60,
          'walking_hallway': 61,
          'walking_walking': 62}

LABELS_NAME = ['desk_work_desk_work',
               'desk_work_napping',
               'desk_work_sit_down',
               'desk_work_stand_up',
               'desk_work_turn_left',
               'desk_work_turn_right',
               'desk_work_writing',
               'driving_accelerate',
               'driving_decelerate',
               'driving_driving',
               'driving_still',
               'driving_stop',
               'driving_turn_left',
               'driving_turn_right',
               'lunch_drinking',
               'lunch_eating',
               'lunch_ordering',
               'lunch_turn_left',
               'lunch_turn_right',
               'office_talk_check_phone',
               'office_talk_office_talk',
               'office_talk_reach',
               'office_talk_turn_left',
               'office_talk_turn_right',
               'ping-pong_bounce_ball',
               'ping-pong_hit',
               'ping-pong_pickup_ball',
               'ping-pong_ping-pong',
               'ping-pong_serve',
               'playing_cards_playing_cards',
               'playing_cards_put_card',
               'playing_cards_shuffle',
               'playing_cards_take_card',
               'playing_pool_chalk_up',
               'playing_pool_check_phone',
               'playing_pool_playing_pool',
               'playing_pool_reach',
               'playing_pool_shooting',
               'playing_pool_turn_left',
               'playing_pool_turn_right',
               'running_looking_at',
               'running_running',
               'running_turn_around',
               'sitting_at_computer',
               'sitting_check_phone',
               'sitting_follow_obj',
               'sitting_reach',
               'sitting_sitting',
               'sitting_turn_left',
               'sitting_turn_right',
               'stairs_doorway',
               'stairs_down_stairs',
               'stairs_reach',
               'stairs_turn_left',
               'stairs_turn_right',
               'stairs_up_stairs',
               'standing_leaning',
               'standing_standing',
               'walking_breezeway',
               'walking_crossing_street',
               'walking_doorway',
               'walking_hallway',
               'walking_walking']

M_NUM_CLASSES = len(M_LABELS_NAME)
S_NUM_CLASSES = len(S_LABELS_NAME)
NUM_CLASSES = len(LABELS_NAME)


IMG_TRANSFORM = T.Compose([lambda x:torch.from_numpy(x).float()/255.0,
                        #    lambda x:x.view(-1,3,x.size(1),x.size(2)).permute(1,0,2,3),
                           T.Normalize(mean=[0.485, 0.456, 0.406]*DEPTH,
                                       std=[0.229, 0.224, 0.225]*DEPTH)
                           ],)

FLO_TRANSFORM = T.Compose([lambda x:torch.from_numpy(x).float()/255.0,
                        #    lambda x:x.view(-1,2,x.size(1),x.size(2)).permute(1,0,2,3),
                           T.Normalize(mean=[0.485, 0.456]*DEPTH,
                                       std=[0.229, 0.224]*DEPTH)
                           ],)


TRAIN_LOADER_ARGS = dict(batch_size = 16,
                         shuffle = True,
                         num_workers = 4,
                         pin_memory = True,
                         drop_last = True,
                         prefetch_factor = 4,
                         persistent_workers = True,
                         )

TEST_LOADER_ARGS = dict(batch_size = 16,
                        shuffle = False,
                        num_workers = 4,
                        pin_memory = True,
                        drop_last = False,
                        prefetch_factor = 4,
                        persistent_workers = True,
                        )
if socket.gethostname() == 'rainbow-panda':   
   DATA_ROOT = "/home/k_b459/DATA/data/"
else:
   DATA_ROOT = "/data/keshav/360/finalEgok360/data/"