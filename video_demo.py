import cv2

import tensorflow as tf
import numpy as np
import C3D_model
import data_processing
TRAIN_LOG_DIR = 'Log/train/'
TRAIN_CHECK_POINT = 'check_point/train.ckpt-49'

BATCH_SIZE = 1
NUM_CLASSES = 2
CROP_SZIE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 16

def main():
    with open('classInd.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    with tf.Graph().as_default():
        batch_clips = tf.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, CROP_SZIE, CROP_SZIE, CHANNEL_NUM], name='X')
	batch_labels = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_CLASSES], name='Y')
    	keep_prob = tf.placeholder(tf.float32)
    	logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)
    	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_labels, 1)), np.float32))
	restorer = tf.train.Saver()
        classifier=tf.nn.softmax(logits) #class
    	restorer = tf.train.Saver()
    	config = tf.ConfigProto()
    	config.gpu_options.allow_growth = True
    	with tf.Session(config=config) as sess:
        	sess.run(tf.global_variables_initializer())
        	sess.run(tf.local_variables_initializer())
        	restorer.restore(sess, TRAIN_CHECK_POINT)
        	accuracy_epoch = 0
        	batch_index = 0
		video = 'videos/Clip1/Clip1_1.mp4'
    		cap = cv2.VideoCapture(video)
		out_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		out_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps=cap.get(cv2.CAP_PROP_FPS)
		fourcc=cv2.VideoWriter_fourcc(*'XVID')
		out=cv2.VideoWriter('out.mp4',fourcc,fps,(out_width,out_height))
		#cap.set(cv2.CAP_PROP_FPS,16)
		print(cap.get(cv2.CAP_PROP_FPS))
    		clip = []
		n=0
    		while True:
        		ret, frame = cap.read()
        		if ret:
            			tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				clip.append(tmp)
            			if len(clip) == 16:
					#print clip.shape
					#print np.asarray(clip).shape
					clip2=data_processing.frame_process(clip,16,112,3)
					#print(clip2.shape)
					newclip=[]
					newclip.append(clip2)
					newclip=np.array(newclip).astype(np.float32)
					#print(newclip.shape)
					classification=sess.run(classifier,feed_dict={batch_clips:newclip,keep_prob: 1.0})
            				label=np.argmax(classification[0])		
					cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (40, 80),cv2.FONT_HERSHEY_SIMPLEX, 3,(0, 0, 255), 3)
					print(class_names[label])
               				clip.pop(0)
            				cv2.imshow('result', frame)
					out.write(frame)
            				cv2.waitKey(10)
				out.write(frame)
				n=n+1
				#print(n)
        		else:
            			break
		cap.release()
		out.release()
		cv2.destroyAllWindows()
'''
video = 'videos/s1_4.mp4'
cap = cv2.VideoCapture(video)
clip = []
label=1
'''
'''
with open('classInd.txt', 'r') as f:
	class_names = f.readlines()
	print class_names[0],class_names[1],class_names[2]
	f.close()  
'''    
'''  
while True:
	ret, frame = cap.read()
	if ret:
		tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		clip.append(tmp)
		if len(clip) == 16:
			print np.asarray(clip).shape
			clip2=data_processing.frame_process(clip,16,112,3)
			print(clip2.shape)

			cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
			
                	cv2.putText(frame, "prob: %.4f" % pred[0][label],(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 1)
               		clip.pop(0)
            	cv2.imshow('result', frame)
            	cv2.waitKey(10)
        else:
            break
cap.release()
cv2.destroyAllWindows()
'''

if __name__ == '__main__':
    main()

               
