from picamzero import Camera

cam = Camera()
cam.start_preview()
cam.take_photo("/home/userk/aa.jpg")
cam.stop_preview()
