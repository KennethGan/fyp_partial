import cv2
import numpy as np

leftlineequation = []
rightlineequation = []
class lineequation():
	
	def getlines(self, lines):
		#leftlineequation = [0, 0]
		#rightlineequation = [0, 0]

		leftlineequation = [0]
		rightlineequation = [0]

		if lines is not None:
			for left_x1, left_y1, left_x2, left_y2 in lines[0]:

				if left_x1 == 0 or left_y1 == 0 or left_x2 == 0 or left_y2 == 0:
					return None
				
				leftlineequation = self.makeequation(left_x1, left_y1, left_x2, left_y2)

				#print("Left Line Equation: y = "+str(leftlineequation[0])+"x +" + str(leftlineequation[1]))

			for right_x1, right_y1, right_x2, right_y2 in lines[1]:

				if right_x1 == 0 or right_y1 == 0 or right_x2 == 0 or right_y2 == 0:
					return None

				rightlineequation = self.makeequation(right_x1, right_y1, right_x2, right_y2)
	
				#print("Right Line Equation: y = "+str(rightlineequation[0])+"x +" + str(rightlineequation[1]))
	    			
		lineequations = [leftlineequation, rightlineequation]
		if lineequations[0][0] == 0 and lineequations[1][0] == 0:
			return None
		return lineequations

	def makeequation(self, x1, y1, x2, y2):
		gradient = (y2 - y1) // (x2 - x1)
		yintercept = y1 - (gradient * x1)
		
		equation = [gradient, yintercept]
		return equation
		
