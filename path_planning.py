
import numpy as np
from scipy.interpolate import CubicSpline, make_splprep, splprep, splev
class PathPlanning:
    def __init__(self):        
        
        # Fixed Starting Point in front of the car
        self.car_position = np.array([48, 68])
        
        self.startup_mode = True
        self.startup_counter = 10  # first "frames" for startup mode so the zoom doesnt create a weird path
        self.facing_vector = np.array([0, -1])  # Default facing vector (up)
        self.consecutive_valid_count = 0
        self.consecutive_valid_threshold = 40
        self.using_facing_mode = False
        # Stage selection (enable/disable different planning strategies)

        # Stage 0 - Startup-Mode = Just go forward in the first frames
        # Stage 0.5 - Emergency-Path - If we go off lane, go back to one lane or search for new lane
        self.just_use_middle_path = False # Stage 1 - Use the middle curve
        self.check_straight_line = False  # Stage 2 - Doesnt work on by itself but can optimize double or wide curves
        self.just_use_vecotrs = True # Stage 3 - Move the vecotrs in curves to the inner lane
        self.number_of_points = 50

        self.desired_distance = 6.5

        #Last known path and curvature 
        self.last_path = None
        self.min_curvature_value = 0.05
        self.last_curvature = self.min_curvature_value

    def plan(self, left_lane_boundary, right_lane_boundary):
        """
        Args:
            left_lane_boundary: Array of [x,y] points defining the left lane boundary
            right_lane_boundary: Array of [x,y] points defining the right lane boundary
            
        Returns:
            way_points: List of [x,y] points defining the planned path
            curvature: Estimate of the path curvature
        """
        try:
            # For the first second, just go straight forward
            if self.startup_mode and not self.just_use_middle_path:
                if self.startup_counter > 0:
                    self.startup_counter -= 1
                    path_end = self.car_position + np.array([0, -1]) * 60              
                    t = np.linspace(0, 1, self.number_of_points)
                    straight_path = np.zeros((self.number_of_points, 2))
                    
                    for i, t_val in enumerate(t):
                        straight_path[i] = self.car_position * (1-t_val) + path_end * t_val
                    
                    self.last_path = straight_path
                    self.last_curvature = self.min_curvature_value

                    straight_path = self.resample_path_uniform_distance(straight_path)               
                    return straight_path, self.min_curvature_value
                else:
                    self.startup_mode = False
                    
            # Emergency handling if we go offroad
            if len(left_lane_boundary) < 2 or len(right_lane_boundary) < 2:
                emergency_path = self.create_emergency_path(left_lane_boundary, right_lane_boundary)
                self.last_path = emergency_path
                self.last_curvature = self.min_curvature_value

                emergency_path = self.resample_path_uniform_distance(emergency_path)
                return emergency_path, self.min_curvature_value

            
            # Convert to numpy arrays for easier handling
            left = np.array(left_lane_boundary)
            right = np.array(right_lane_boundary)
            
            # Determine how many points exist in both arrays
            min_lane_boundary = min(len(left), len(right))
            left_points = left[:min_lane_boundary]
            right_points = right[:min_lane_boundary]

            middle_points = (left_points + right_points) / 2
            forward_mask = middle_points[:, 1] < self.car_position[1]
            middle_points = middle_points[forward_mask]
            # Stage 1: Only use Middle Points in front of the car
            if(self.just_use_middle_path):
                curvature = self.calculate_curvature(middle_points)
                x_middle, y_middle = middle_points[:, 0], middle_points[:, 1]
                x_middle, y_middle, t = self.fix_path_data(x_middle, y_middle)
                middle_points = np.column_stack((x_middle, y_middle))

                middle_points = self.resample_path_uniform_distance(middle_points)
                self.last_path = middle_points
                self.last_curvature = curvature
                return middle_points, curvature
            
            
            # STAGE 2: Check if straight line is valid
            if self.check_straight_line and len(left_points) >= 2:
                straight_path = self.check_straight_line_validity(left_points, right_points)
                if straight_path is not None:
                    full_path = np.vstack([self.car_position, straight_path])
                    curvature = max(self.calculate_curvature(full_path), self.min_curvature_value)
                
                    
                    self.last_path = full_path
                    self.last_curvature = curvature
                    full_path = self.resample_path_uniform_distance(full_path)
                    return full_path, curvature
                
            # Stage 3: Use Vectors and move vectors inside the lane
            if self.just_use_vecotrs:  
                curvature = self.calculate_curvature(middle_points)

                middle_points = np.array(middle_points)


                # Make it to a spline back agin to limit the points to num points to better calculate stuff
                x_middle, y_middle = middle_points[:, 0], middle_points[:, 1]

                if len(x_middle) < 4:
                    raise ValueError("Nicht genug Punkte für splprep.")

                if np.any(np.isnan(x_middle)) or np.any(np.isnan(y_middle)):
                    raise ValueError("NaN in den Punkten!")

                if np.allclose(x_middle, x_middle[0]) and np.allclose(y_middle, y_middle[0]):
                    raise ValueError("Alle Punkte sind gleich – kein gültiger Pfad.")
                
                x_middle, y_middle, t = self.fix_path_data(x_middle, y_middle)

                middle_tck, u = splprep([x_middle, y_middle], u=t, s=3.41)
                u_middle = np.linspace(0, 1, num=18)
                x_new, y_new = splev(u_middle, middle_tck)
                middle_points = np.column_stack((x_new, y_new))



                if len(middle_points) >= 3:
                    # Calculate overall direction of the path (first to last point)
                    main_direction = middle_points[-1] - middle_points[0]
                    main_direction = main_direction / np.linalg.norm(main_direction)
                    
                    # Check if path is mostly straight by comparing each segment to main direction
                    is_straight_path = True
                    deviation_threshold = 0.145  # Adjust based on testing
                    
                    for i in range(len(middle_points)-1):
                        segment = middle_points[i+1] - middle_points[i]
                        if np.linalg.norm(segment) > 0:  # Avoid zero-length segments
                            segment = segment / np.linalg.norm(segment)
                            # Dot product close to 1 means segments are aligned with main direction
                            alignment = np.abs(np.dot(segment, main_direction))
                            if alignment < (1 - deviation_threshold):
                                is_straight_path = False
                                break
                    
                    if is_straight_path:
                        # If the path is straight, don't modify points based on curvature
                        curvature = self.calculate_curvature(middle_points)
                        middle_points = self.resample_path_uniform_distance(middle_points)
                        return middle_points, curvature


                dx_dt = np.gradient(middle_points[:, 0])
                dy_dt = np.gradient(middle_points[:, 1])
                velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
                ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
                tangent = np.array([1/ds_dt]).transpose() * velocity
                tangent_x = tangent[:, 0]
                tangent_y = tangent[:, 1]

                deriv_tangent_x = np.gradient(tangent_x)
                deriv_tangent_y = np.gradient(tangent_y)
                dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
                length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
                normal = np.array([1/length_dT_dt]).transpose() * dT_dt

                
                # Alternative simpler condition
                for i in range(1, len(middle_points)-1):
                    # Calculate angle between vectors from this point to neighbors
                    v1 = middle_points[i-1] - middle_points[i]
                    v2 = middle_points[i+1] - middle_points[i]
                    
                    # Normalize vectors
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Calculate angle between vectors
                    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                    
                    # Only adjust points at significant angles
                    if angle > 0.01:  # About 5.7 degrees
                        middle_points[i] = middle_points[i] + -normal[i] * velocity[i]*2.5

                middle_points = self.detect_and_fix_zigzag(middle_points, angle_threshold=0.5)

                removedPoints = 0
                for i in range(3,len(middle_points)-1 - removedPoints):
                    middle_points = np.array(middle_points)
                    A = middle_points[i-1]
                    B = middle_points[i]

                    length = np.linalg.norm(B - A)
                    #print(length)
                    if length > 1:
                        np.delete(middle_points, i)
                        removedPoints += 1

                middle_points = self.resample_path_uniform_distance(middle_points)
                curvature = self.calculate_curvature(middle_points)
                self.last_path = middle_points
                self.last_curvature = curvature

                return middle_points, curvature

            final_path = self.resample_path_uniform_distance(full_path)

        except Exception as e:
            print(f"Returning last path and curvature due to unexpected Error: {e}")
            return self.last_path, self.last_curvature

        self.last_path = final_path
        self.last_curvature = curvature
        return final_path, curvature
    
    def detect_and_fix_zigzag(self, path, angle_threshold):
        """
        Detect zigzag patterns in the path and smooth them out.
        
        Args:
            path: numpy array of shape (n, 2) containing x,y coordinates
            angle_threshold: threshold angle (in radians) to detect zigzags
        
        Returns:
            Corrected path as numpy array
        """
        if len(path) < 3:
            return path
        
        corrected_path = np.copy(path)
        
        # Look for zigzag patterns
        for i in range(1, len(path) - 1):
            # Calculate vectors to previous and next points
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate angle between vectors
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # If angle is greater than threshold, we have a zigzag
                if angle > angle_threshold:
                    # Replace the point with the average of its neighbors
                    corrected_path[i] = (path[i-1] + path[i+1]) / 2
        
        return corrected_path

    def resample_path_uniform_distance(self, points):
        """
        Resamples a path so that points are spaced at a uniform distance.
        
        Args:
            points: numpy array of shape (n, 2) containing x,y coordinates
            desired_distance: the desired distance between consecutive points
            
        Returns:
            numpy array of shape (m, 2) with uniformly spaced points
        """

        desired_distance = self.desired_distance
        # Calculate the cumulative distance along the path
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_lengths[-1]
        
        # Calculate how many points we need
        num_points = max(int(total_length / desired_distance) + 1, 4)
        
        # Create a spline of the path
        # Use the cumulative lengths as the parameter to ensure accuracy along curves
        tck, u = splprep([points[:, 0], points[:, 1]], u=cumulative_lengths/total_length, s=0)
        
        # Sample the spline at uniform distances
        uniform_u = np.linspace(0, 1, num_points)
        uniform_points = np.column_stack(splev(uniform_u, tck))
        
        return uniform_points

    def fix_path_data(self, x_middle, y_middle):
        """
        Fix common issues in path data that might cause splprep to fail:
        1. Replace duplicate points with virtual interpolated points
        2. Fix non-monotonic sections using custom parameterization
        """
        # Convert to numpy arrays if not already
        x = np.array(x_middle)
        y = np.array(y_middle)
        
        # Create arrays for our corrected path
        x_fixed = [x[0]]
        y_fixed = [y[0]]
        
        # Small offset to use for virtual points (adjust as needed for your scale)
        offset = 1e-6
        virtual_count = 0
        
        # Process points
        for i in range(1, len(x)):
            # Check if current point is identical to the previous one
            if np.isclose(x[i], x[i-1]) and np.isclose(y[i], y[i-1]):
                # Instead of skipping, create a virtual point with a tiny offset
                # We'll try to create a point that maintains the path direction
                
                # Check if we can determine direction from nearby points
                if i > 1 and i < len(x) - 1:
                    # Use points before and after to determine direction
                    dx_before = x[i-1] - x[i-2]
                    dy_before = y[i-1] - y[i-2]
                    dx_after = x[i+1] - x[i]
                    dy_after = y[i+1] - y[i]
                    
                    # Average direction
                    dx_avg = (dx_before + dx_after) / 2
                    dy_avg = (dy_before + dy_after) / 2
                    
                    # Normalize and scale by offset
                    magnitude = np.sqrt(dx_avg**2 + dy_avg**2)
                    if magnitude > 1e-10:  # Avoid division by zero
                        dx_avg = dx_avg / magnitude * offset
                        dy_avg = dy_avg / magnitude * offset
                    else:
                        # If direction can't be determined, use a default offset
                        dx_avg, dy_avg = offset, offset
                else:
                    # For edge cases, use a default offset
                    dx_avg, dy_avg = offset, offset
                
                # Create virtual point
                x_fixed.append(x[i] + dx_avg)
                y_fixed.append(y[i] + dy_avg)
                virtual_count += 1
            else:
                # Non-duplicate point, add directly
                x_fixed.append(x[i])
                y_fixed.append(y[i])
        
        # Convert lists back to numpy arrays
        x_fixed = np.array(x_fixed)
        y_fixed = np.array(y_fixed)
        
        # Print info about created virtual points
        if virtual_count > 0:
            print(f"Created {virtual_count} virtual points to replace duplicates")
        
        # Calculate a custom parameterization for the fixed path
        t = np.zeros(len(x_fixed))
        t[1:] = np.sqrt(np.diff(x_fixed)**2 + np.diff(y_fixed)**2)
        t = np.cumsum(t)
        
        # Avoid division by zero if all points are identical
        if t[-1] > 0:
            t /= t[-1]  # normalize to [0, 1]
        
        return x_fixed, y_fixed, t

    def update_facing_vector(self, end_point):
        """
        Update the facing vector based on the end point.
        
        Args:
        end_point (np.ndarray): The end point of the path, shape (2,)
        """
        # Calculate direction from car position to end point
        direction = end_point - self.car_position
        
        # Normalize direction to get unit vector
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            self.facing_vector = direction / direction_norm
            print(f"Updated facing vector: {self.facing_vector}")
    
    def check_straight_line_validity(self, left_points, right_points):
        """
        Check if a straight line path from the car's position is valid.
        
        Args:
        left_points (np.ndarray): Points defining left lane boundary, shape (n, 2)
        right_points (np.ndarray): Points defining right lane boundary, shape (n, 2)
        
        Returns:
        np.ndarray or None: Valid path points or None if invalid
        """
        if len(left_points) < 2 or len(right_points) < 2:
            print("Not enough points in lane boundaries")
            self.consecutive_valid_count = 0
            self.using_facing_mode = False
            return None
        
        # Trim initial points
        left_points = left_points[20:]
        right_points = right_points[20:]    
        
        # Get furthest visible point in middle of path
        end_point = (left_points[-1] + right_points[-1]) / 2
        
        # Decide whether to use car-to-end-point path or facing vector path
        if not self.using_facing_mode:
            # Create straight line path from car to end point (center path)
            t = np.linspace(0, 1, len(left_points))
            straight_path = np.outer(1-t, self.car_position) + np.outer(t, end_point)
            print(f"Using center path mode. Consecutive valid: {self.consecutive_valid_count}")
        else:
            # Use facing vector to determine path
            # Calculate the maximum distance we can see
            max_distance = np.linalg.norm(end_point - self.car_position)
            t = np.linspace(0, max_distance, len(left_points))
            straight_path = np.array([self.car_position + i * self.facing_vector for i in t])
            print(f"Using facing vector mode: {self.facing_vector}")
        
        # Check if path is valid (all points inside lane)
        all_valid = True
        for i in range(len(straight_path)):
            # Create line segments between lane boundary points
            if i < len(left_points) - 1 and i < len(right_points) - 1:
                left_segment = np.array([left_points[i], left_points[i+1]])
                right_segment = np.array([right_points[i], right_points[i+1]])
               
                # Check if path point is inside the lane segment
                point = straight_path[i]
                in_lane = self.is_point_inside_lane(point, left_segment, right_segment)
               
                if not in_lane:
                    print(f"Point {i} is outside lane")
                    all_valid = False
                    break
        
        # Update tracking based on validity
        if all_valid:
            if not self.using_facing_mode:
                self.consecutive_valid_count += 1
                if self.consecutive_valid_count >= self.consecutive_valid_threshold:
                    # Switch to facing mode and update facing vector
                    self.using_facing_mode = True
                    self.update_facing_vector(end_point)
                    print("Switching to facing mode")
            
            return straight_path[1:]  # Return valid path points
        else:
            # Reset on invalid path
            self.consecutive_valid_count = 0
            self.using_facing_mode = False
            return None
            
    def is_point_inside_lane(self, point, left_segment, right_segment):
        """
        Check if a point is inside the lane using a precise coordinate system check.
        
        Args:
        point (np.ndarray): The point to check, shape (2,)
        left_segment (np.ndarray): Left lane segment, shape (2, 2)
        right_segment (np.ndarray): Right lane segment, shape (2, 2)
        
        Returns:
        bool: True if point is inside the lane, False otherwise
        """
        # Extract coordinates
        point_x, point_y = point
        left_x1, left_y1 = left_segment[0]
        left_x2, left_y2 = left_segment[1]
        right_x1, right_y1 = right_segment[0]
        right_x2, right_y2 = right_segment[1]
        
        # Debugging print
        print('JETZT', point_x, point_y, point, left_segment, right_segment)
        
        # Looking at the debug output, the correct check is:
        # The point x should be greater than the leftmost x boundary
        # The point x should be less than the rightmost x boundary
        # The lane seems to run left-to-right
        
        # Get the maximum x of the left segment (rightmost point of left boundary)
        max_left_x = max(left_x1, left_x2)
        
        # Get the minimum x of the right segment (leftmost point of right boundary)
        min_right_x = min(right_x1, right_x2)
        
        # Check if the point is between the left and right lane boundaries
        # Point should be to the right of left boundary and to the left of right boundary
        return max_left_x <= point_x <= min_right_x

    def calculate_curvature(self, trajectory):
        """
        Calculate approximate curvature of the path.
        """
        if trajectory is None or len(trajectory) < 3:
            curvature = 0.0           
        # Calculate curvature
        else:
            p1, p2, p3 = map(np.array, [trajectory[0], trajectory[len(trajectory) // 2], trajectory[-1]])
            vec1, vec2 = p2 - p1, p3 - p2
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

            if norm1 > 1e-4 and norm2 > 1e-4:
                dot = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
                curvature = np.arccos(dot)
            else:
                curvature = 0.0
                    
        return curvature
    
    def create_emergency_path(self, left, right):
        """
        Creates an emergency recovery path when lane detection fails or car is off-track.
        """
        left_valid = len(left) > 0 and left[-1] is not None
        right_valid = len(right) > 0 and right[-1] is not None
        if left_valid and right_valid:
            end_point = (left[-1] + right[-1]) / 2
        elif not left_valid and right_valid:
            end_point = right[-1]
        elif left_valid and not right_valid:
            end_point = left[-1]
        else:
            end_point = self.car_position + np.array([0, -1]) * 20

        t = np.linspace(0, 1, self.number_of_points)
        emergency_path = np.zeros((self.number_of_points, 2))        
        for i, t_val in enumerate(t):
            emergency_path[i] = self.car_position * (1-t_val) + end_point * t_val
        
        return emergency_path