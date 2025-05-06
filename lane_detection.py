import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

class LaneDetection:

	def __init__(self):
		self.frame_counter = 0
		self.debug_image = np.zeros((10, 10, 3), dtype=np.uint8)
		pass

	def detect(self, state_image):
		self.frame_counter += 1
		if self.frame_counter < 60:
			return [], []

		masked_img = state_image[0:84, ...]
		binary_img = self.convert_to_binary(masked_img)

		# Convert to grayscale and preprocess
		filtered = self.filterEdges(binary_img)
		self.applyTreshold(filtered, np.percentile(filtered, 80))

		# Extract edge points
		edge_points = np.column_stack(np.where(filtered == 255))

		# switch x and y coordinates converting to correct coordinate system
		edge_points = np.array([(point[1], point[0]) for point in edge_points])

		# Use DBSCAN for clustering
		dbscan = DBSCAN(eps=4, min_samples=1).fit(edge_points)  # eps: how close points need to be to be considered in the same cluster
		labels = dbscan.labels_

		# Separate points into clusters
		unique_labels = set(labels)
		clusters = {label: edge_points[labels == label] for label in unique_labels if label != -1}  # Exclude noise (-1)

		 # Sort clusters by size (number of points)
		sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

		# Take the two largest clusters
		if len(sorted_clusters) < 2:
			print("Not enough clusters found to identify left and right lanes.")
			return [], []
		largest_clusters = sorted_clusters[:2]

		# differentiate left and right
		lane_1 = largest_clusters[0][1].astype(np.float64)
		lane_2 = largest_clusters[1][1].astype(np.float64)

		# Calculate the center of the two clusters
		lane_1_center = np.mean(lane_1, axis=0)
		lane_2_center = np.mean(lane_2, axis=0)
		if lane_1_center[0] < lane_2_center[0]:
			left_points = lane_1
			right_points = lane_2
		else:
			left_points = lane_2
			right_points = lane_1

		# Sort points by y-coordinate (bottom to top)
		left_points = (left_points[np.argsort(left_points[:, 1])])[::-1]
		right_points = (right_points[np.argsort(right_points[:, 1])])[::-1]

		left_points = left_points.astype(np.int32)
		right_points = right_points.astype(np.int32)
		debug_image = np.zeros((filtered.shape[0], filtered.shape[1], 3), dtype=np.uint8)

		# color left and right lane points, left is blue, right is red
		for point in left_points:
			if 0 < point[0] < 96 and 0 < point[1] < 84:
				debug_image[int(point[1]), int(point[0])] = [255, 0, 0]
		for point in right_points:
			if 0 < point[0] < 96 and 0 < point[1] < 84:
				debug_image[int(point[1]), int(point[0])] = [0, 0, 255]

		self.debug_image = debug_image

		return left_points, right_points

	def convert_to_binary(self, image):
		# get all curb points (color 255,0,0 or 255,255,255)
		curb_points = np.argwhere(np.all(image == [255, 0, 0], axis=-1))
		np.append(curb_points, np.argwhere(np.all(image == [255, 255, 255], axis=-1)))

		# Reshape the image into a 2D array of pixels (rows * cols, channels)
		pixels = image.reshape(-1, image.shape[-1])

		# Apply KMeans clustering with 2 clusters
		kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
		labels = kmeans.labels_

		# Reshape the labels back to the original image shape
		binary_image = labels.reshape(image.shape[:2])

		# Map the two clusters to 0 and 255
		binary_image = (binary_image * 255).astype(np.uint8)

		# set all curb points to 1
		mean = np.mean(binary_image)
		for point in curb_points:
			if 0 < point[0] < 96 and 0 < point[1] < 84:
				binary_image[int(point[0]), int(point[1])] = 1 if mean < 0.5 else 0

		return binary_image

	def filterEdges(self, image):
		# Sobel kernels
		sobel_x = np.array([[1, 0, -1],
							[2, 0, -2],
							[1, 0, -1]])
		sobel_y = np.array([[1, 2, 1],
							[0, 0, 0],
							[-1, -2, -1]])

		# Apply Sobel filters
		grad_x = self.convolve2d(image, sobel_x)
		grad_y = self.convolve2d(image, sobel_y)

		# combine gradients by distance
		magnitude = np.sqrt(grad_x**2 + grad_y**2)
		
		# mask the car and then normalize to 0-255
		magnitude[64:78, 44:52] = 0
		normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255

		return normalized[1:-1, 1:-1]  # Remove one pixel from each side to avoid border effects

	# apply kernel on 2d image, kernel can by any size, has to be quadratic with uneven sidelength
	def convolve2d(self, image, kernel):
		pad_width = kernel.shape[0] // 2
		padded_image = np.pad(image, pad_width, mode='mean')  # Zero-padding
		
		output = np.zeros_like(image)
		for y in range(image.shape[0]):
			for x in range(image.shape[1]):
				output[y, x] = np.sum(padded_image[y:y+kernel.shape[0], x:x+kernel.shape[1]] * kernel)
		return output

	def applyTreshold(self, img, threshold):
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if img[i, j] > threshold:
					img[i, j] = 255
				else:
					img[i, j] = 0