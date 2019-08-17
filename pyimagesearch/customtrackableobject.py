class CarTrackableObject:
	def __init__(self, carobjectID, carcentroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.carobjectID = carobjectID
		self.carcentroids = [carcentroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

class BusTrackableObject:
	def __init__(self, busobjectID, buscentroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.busobjectID = busobjectID
		self.buscentroids = [buscentroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

class BikeTrackableObject:
	def __init__(self, bikeobjectID, bikecentroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.bikeobjectID = bikeobjectID
		self.bikecentroids = [bikecentroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
