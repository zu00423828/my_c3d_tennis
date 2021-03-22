import cv2

video = 'videos/Clip1/Clip1_1.mp4'
cap = cv2.VideoCapture(video)
clip = []
label=1
'''
'''
with open('classInd.txt', 'r') as f:
	class_names = f.readlines()
	print class_names[0],class_names[1],class_names[2]
	f.close()  
while True:
	ret, frame = cap.read()
	if ret:
            	cv2.imshow('result', frame)
            	cv2.waitKey(0)
		print('123')
		
        else:
            break
cap.release()
cv2.destroyAllWindows()
