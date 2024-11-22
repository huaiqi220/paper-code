from model import CGES
import config

ddp_model = CGES.mobile_gaze_2d(config.hm_size, 12, 25 * 25)