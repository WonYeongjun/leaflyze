from picamzero import Camera

cam = Camera()
cam.start_preview()
cam.take_photo("/home/userk/new/newimg45green.jpg")
cam.stop_preview()