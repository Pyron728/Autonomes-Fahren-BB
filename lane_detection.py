import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

class LaneDetection:

	def __init__(self):
		self.frame_counter = 0
		self.debug_image = np.zeros((10, 10, 3), dtype=np.uint8)
		self.last_left_points = []
		self.last_right_points = []
		pass

	def detect(self, state_image):
		self.frame_counter += 1
		if self.frame_counter < 60:
			return self.last_left_points, self.last_right_points

		masked_img = state_image[0:84, ...]
		binary_img = self.convert_to_binary(masked_img)

		# Convert to grayscale and preprocess
		filtered = self.filterEdges(binary_img)
		self.applyTreshold(filtered, np.percentile(filtered, 80))

		# Extract edge points
		edge_points = np.column_stack(np.where(filtered == 255))

		# switch x and y coordinates converting to correct coordinate system
		edge_points = np.array([(point[1], point[0]) for point in edge_points])

		if len(edge_points) < 0 or edge_points.shape[0] == 0:
			return self.last_left_points, self.last_right_points

		# Use DBSCAN for clustering
		dbscan = DBSCAN(eps=3, min_samples=1).fit(edge_points)  # eps: how close points need to be to be considered in the same cluster
		labels = dbscan.labels_

		# Separate points into clusters
		unique_labels = set(labels)
		clusters = {label: edge_points[labels == label] for label in unique_labels if label != -1}  # Exclude noise (-1)

		 # Ignore clusters with less than 15 points
		clusters = {label: points for label, points in clusters.items() if len(points) > 15}

		# Compute closest points to the car position for each cluster
		car_pos = np.array([48, 64])
		sorted_clusters = []
		for label, points in clusters.items():
			# Calculate the distance from the car position to the cluster points
			distances = np.linalg.norm(points - car_pos, axis=1)
			# Find the closest point in the cluster to the car position
			closest_point_index = np.argmin(distances)
			sorted_clusters.append((distances[closest_point_index], points))

		# Take the two largest clusters with the closest points to the car position
		sorted_clusters.sort(key=lambda x: x[0])  # Sort by distance
		if len(sorted_clusters) < 2:
			# print("Not enough clusters found.")
			return self.last_left_points, self.last_right_points

		# differentiate left and right
		lane_1 = sorted_clusters[0][1].astype(np.float64)
		lane_2 = sorted_clusters[1][1].astype(np.float64)

		# Calculate the center of the two clusters
		lane_1_center = np.mean(lane_1, axis=0)
		lane_2_center = np.mean(lane_2, axis=0)
		if lane_1_center[0] < lane_2_center[0]:
			left_points = lane_1
			right_points = lane_2
		else:
			left_points = lane_2
			right_points = lane_1

		left_points = self.sort_edge_points(car_pos, left_points)
		right_points = self.sort_edge_points(car_pos, right_points)

		left_points = left_points.astype(np.int32)
		right_points = right_points.astype(np.int32)

		debug_image = np.zeros((filtered.shape[0], filtered.shape[1], 3), dtype=np.uint8)
		debug_image = state_image[0:filtered.shape[0], 0:filtered.shape[1], :]
		# color left and right lane points, left is blue, right is red
		i = 0
		for point in left_points:
			if 0 < point[0] < 96 and 0 < point[1] < 84:
				debug_image[int(point[1]), int(point[0])] = [255, int(i), 0]
				i += 255 / len(left_points)
		i = 0
		for point in right_points:
			if 0 < point[0] < 96 and 0 < point[1] < 84:
				debug_image[int(point[1]), int(point[0])] = [0, int(i), 255]
				i += 255 / len(right_points)

		# seperatly mark first and last point of the lanes
		if len(left_points) > 0:
			debug_image[int(left_points[0][1]), int(left_points[0][0])] = [255, 255, 255]
			debug_image[int(left_points[-1][1]), int(left_points[-1][0])] = [255, 255, 255]
		if len(right_points) > 0:
			debug_image[int(right_points[0][1]), int(right_points[0][0])] = [255, 255, 255]
			debug_image[int(right_points[-1][1]), int(right_points[-1][0])] = [255, 255, 255]

		self.debug_image = debug_image

		self.last_left_points = left_points
		self.last_right_points = right_points
		return left_points, right_points

	def convert_to_binary(self, image):
		# get all curb points (color 255,0,0 or 255,255,255)
		curb_colors = np.array([[253, 0, 0], [253, 9, 9], [253, 17, 17], [253, 253, 252], [253, 253, 253], [253, 251, 251]])
		
		# get array of point coordinates for the points where the color is in curb_colors
		curb_points = np.array(np.where(np.all(image[:, :, :3] == curb_colors[:, None, None], axis=0))).T

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
				x = int(point[0])
				y = int(point[1])
				binary_image[x-1:x+1, y-1:y+1] = 1 if mean < 0.5 else 0

		return binary_image

	def sort_edge_points(self, car_pos, edge_points):
		if (len(edge_points) == 0):
			return np.array([])
		
		# get points at bottom line of the image
		lowest_y_val = edge_points[np.argmax(edge_points[:, 1]), 1]
		bottom_line = edge_points[edge_points[:, 1] == lowest_y_val]

		# find nearest point to car position
		start_pos_index = np.argmin(np.linalg.norm(bottom_line - car_pos, axis=1))
		start_point = bottom_line[start_pos_index]
		sorted_points = [start_point]
		remaining_points = edge_points.tolist()
		remaining_points.remove(start_point.tolist())

		current_point = start_point
		while remaining_points:
			# Find the nearest point to the current point
			distances = np.linalg.norm(np.array(remaining_points) - current_point, axis=1)
			nearest_index = np.argmin(distances)
			
			# avoid points that jump back to a line that is already sorted
			if (distances[nearest_index] > 10):
				break

			nearest_point = remaining_points.pop(nearest_index)

			# Add the nearest point to the sorted list
			sorted_points.append(nearest_point)
			current_point = nearest_point

		return np.array(sorted_points)

	def filterEdges(self, image):
		# laplacian edge detection kernel
		kernel = np.array([[-1, -1, -1],
							[-1, 8, -1],
							[-1, -1, -1]])
		
		magnitude = self.convolve2d(image, kernel)
		
		# erosion to remove noise and small edges
		magnitude = self.erosion(magnitude)
		
		# mask the car and then normalize to 0-255
		magnitude[64:78, 44:52] = 0

		norm_val = np.max(magnitude) - np.min(magnitude)
		if (norm_val == 0):
			# set every value that is not 0 to 255
			normalized = np.zeros_like(magnitude)
			normalized[magnitude != 0] = 255
		else:
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

	# do a 2x2 erosion on the image
	def erosion(self, image):
		output = np.zeros_like(image)
		for i in range(0, image.shape[0] - 1):
			for j in range(0, image.shape[1] - 1):
				if image[i, j] != 0 and image[i + 1, j] != 0 and image[i + 1, j + 1] != 0 and image[i, j + 1] != 0:
					output[i+1, j+1] = image[i, j] / 4 + image[i + 1, j] / 4 + image[i + 1, j + 1] / 4 + image[i, j + 1] / 4
				else:
					output[i+1, j+1] = 0
		return output

	def applyTreshold(self, img, threshold):
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if img[i, j] > threshold:
					img[i, j] = 255
				else:
					img[i, j] = 0