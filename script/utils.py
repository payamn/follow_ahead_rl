import math

def wrap_pi_to_pi(angle):
  while angle > math.pi:
    angle -= math.pi
  while angle < - math.pi:
    angle += 2*math.pi
  return angle

def to_image_coordinate(pos, center_pos, res=(500,500)):
    return (int((pos[0] - center_pos[0])*res[0]/2/5+res[0]/2), int((pos[1] - center_pos[1])*res[1]/2/5+res[1]/2))
