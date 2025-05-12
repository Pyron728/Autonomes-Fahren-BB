
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
        # Stage 0.5 - Emergency-Path - If we go off lane, go bback to one lane or search for new lane
        self.just_use_middle_path = True # Stage 1 - Use the middle curve
        self.check_straight_line = True  # Stage 2 - Doesnt work on by itself but can optimize double or wide curves
        self.just_use_vecotrs = True # Stage 3 - Move the vecotrs in curves to the inner lane

        self.other_splines = False # Stage 4 - Spline creation with the Apex-point to maximise the radius

        self.number_of_points = 20 # How many points should get returend, the more points the lower the curvature 

        # Stage 3 Parameters/Storage Variables
        self.apex_history = [] # 
        self.max_history_length = 5  # Reduced from 10 for faster response
        self.curvature_threshold = 0.001  # Reduced from 0.001      
        self.min_curve_idx = 30  # Reduced from 30
        self.lookahead_distance = 50  # How far ahead to analyze for upcoming curves    
        self.future_weight = 0.5 # Wiehgt of the predicted curves
        self.min_curvature_value = 0.05  # Ensures non-zero curvature for steering  

        #Last known path and curvature for smoothing transitions
        self.last_path = None
        self.last_curvature = self.min_curvature_value
        self.path_smoothing_factor = 0.3  # How much to blend with previous path

    def plan(self, left_lane_boundary, right_lane_boundary):
        """
        Args:
            left_lane_boundary: Array of [x,y] points defining the left lane boundary
            right_lane_boundary: Array of [x,y] points defining the right lane boundary
            
        Returns:
            way_points: List of [x,y] points defining the planned path
            curvature: Estimate of the path curvature
        """
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
                return straight_path, self.min_curvature_value
            else:
                self.startup_mode = False
                
        # Emergency handling if we go offroad
        if len(left_lane_boundary) < 2 or len(right_lane_boundary) < 2:
            emergency_path = self.create_emergency_path(left_lane_boundary, right_lane_boundary)
            self.last_path = emergency_path
            self.last_curvature = self.min_curvature_value
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
            return middle_points, curvature
        
        
        # STAGE 2: Check if straight line is valid
        if self.check_straight_line and len(left_points) >= 2:
            straight_path = self.check_straight_line_validity(left_points, right_points)
            if straight_path is not None:
                full_path = np.vstack([self.car_position, straight_path])
                curvature = max(self.calculate_curvature(full_path), self.min_curvature_value)
              
                
                self.last_path = full_path
                self.last_curvature = curvature
                return full_path, curvature
            
        # Stage 3: Use Vectors and move vectors inside the lane
        if self.just_use_vecotrs:  
            forward_mask = middle_points[:, 1] < self.car_position[1]
            middle_points = middle_points[forward_mask]
            curvature = self.calculate_curvature(middle_points)

            new_path = np.array(middle_points)

            # Make it to a spline back agin to limit the points to num points to better calculate stuff
            x_middle, y_middle = new_path[:, 0], new_path[:, 1]

            if len(x_middle) < 4:
                raise ValueError("Nicht genug Punkte für splprep.")

            if np.any(np.isnan(x_middle)) or np.any(np.isnan(y_middle)):
                raise ValueError("NaN in den Punkten!")

            if np.allclose(x_middle, x_middle[0]) and np.allclose(y_middle, y_middle[0]):
                raise ValueError("Alle Punkte sind gleich – kein gültiger Pfad.")
            
            x_middle, y_middle, t = self.fix_path_data(x_middle, y_middle)

            middle_tck, u = splprep([x_middle, y_middle], u=t, s=1e-3)
            u_middle = np.linspace(0, 1, num=18)
            x_new, y_new = splev(u_middle, middle_tck)
            middle_points = np.column_stack((x_new, y_new))
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
            d2s_dt2 = np.gradient(ds_dt)
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
            t_component = np.array([d2s_dt2]).transpose()
            n_component = np.array([curvature * ds_dt * ds_dt]).transpose()

            acceleration = t_component * tangent + n_component * normal 
            for i in range(len(middle_points)-1):
                if abs(t_component[i]) > 1e-10:
                    middle_points[i] = middle_points[i] + -normal[i] * velocity[i]*2


            counter = 0
            for i in range(3,len(middle_points)-1 - counter):
                middle_points = np.array(middle_points)
                A = middle_points[i-1]
                B = middle_points[i]

                length = np.linalg.norm(B - A)
                #print(length)
                if length > 8:
                    np.delete(middle_points, i)
                    counter += 1
            # Convert final path to number of points in the parameters
            new_path = middle_points

            curvature = self.calculate_curvature(middle_points)
            return middle_points, curvature

        # Stage 4:  Use a Spline to the apex to maximise the radius
        if self.other_splines:
            # Prepare splines for left and right boundaries with caution to not crash it if we have insufficient points
            try:
                # Minimum required points for splprep
                if len(left) < 4 or len(right) < 4:
                    # Fall back to middle path with no apex
                    final_path = np.vstack([self.car_position, middle_points])
                    curvature = max(self.calculate_curvature(final_path), self.min_curvature_value)
                    
                    # Apply path smoothing with previous path
                    if self.last_path is not None and len(self.last_path) == len(final_path):
                        final_path = self.last_path * self.path_smoothing_factor + final_path * (1 - self.path_smoothing_factor)
                        curvature = self.last_curvature * self.path_smoothing_factor + curvature * (1 - self.path_smoothing_factor)
                    
                    self.last_path = final_path
                    self.last_curvature = curvature
                    return final_path, curvature
                    
                tck_left, _ = splprep(left.T, s=0)
                tck_right, _ = splprep(right.T, s=0)
            except TypeError as e:
                # Handle "m > k must hold" error
                if "m > k must hold" in str(e):
                    # Fall back to simple center line path
                    final_path = np.vstack([self.car_position, middle_points])
                    curvature = max(self.calculate_curvature(final_path), self.min_curvature_value)
                    
                    # Apply path smoothing with previous path
                    if self.last_path is not None and len(self.last_path) == len(final_path):
                        final_path = self.last_path * self.path_smoothing_factor + final_path * (1 - self.path_smoothing_factor)
                        curvature = self.last_curvature * self.path_smoothing_factor + curvature * (1 - self.path_smoothing_factor)
                    
                    self.last_path = final_path
                    self.last_curvature = curvature
                    return final_path, curvature
            except Exception as e:
                # General error handling for any spline creation issues
                print(f"Spline error: {e}")
                final_path = np.vstack([self.car_position, middle_points])
                curvature = max(self.calculate_curvature(final_path), self.min_curvature_value)
                
                # Apply path smoothing with previous path
                if self.last_path is not None and len(self.last_path) == len(final_path):
                    final_path = self.last_path * self.path_smoothing_factor + final_path * (1 - self.path_smoothing_factor)
                    curvature = self.last_curvature * self.path_smoothing_factor + curvature * (1 - self.path_smoothing_factor)
                
                self.last_path = final_path
                self.last_curvature = curvature
                return final_path, curvature
            
            # Generate parameter values for spline evaluation
            u = np.linspace(0, 1, 100)
            
            # Evaluate splines
            left_curve = np.array(splev(u, tck_left)).T
            right_curve = np.array(splev(u, tck_right)).T
            
            # Calculate first derivatives
            left_deriv1 = np.array(splev(u, tck_left, der=1)).T
            right_deriv1 = np.array(splev(u, tck_right, der=1)).T
            
            # Calculate second derivatives
            left_deriv2 = np.array(splev(u, tck_left, der=2)).T
            right_deriv2 = np.array(splev(u, tck_right, der=2)).T
            
            # Calculate curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
            left_curvature = np.abs(left_deriv1[:,0] * left_deriv2[:,1] - left_deriv1[:,1] * left_deriv2[:,0]) / \
                            np.power(left_deriv1[:,0]**2 + left_deriv1[:,1]**2, 1.5)
            
            right_curvature = np.abs(right_deriv1[:,0] * right_deriv2[:,1] - right_deriv1[:,1] * right_deriv2[:,0]) / \
                            np.power(right_deriv1[:,0]**2 + right_deriv1[:,1]**2, 1.5)
            
            # Filter out small curvature values in straight lines
            left_curvature[left_curvature < self.curvature_threshold] = 0
            right_curvature[right_curvature < self.curvature_threshold] = 0
            middle_curve = (left_curve + right_curve) / 2 
            vectors = np.zeros((len(middle_curve)-1, 2))
            for i in range(len(middle_curve)-1):
                vectors[i] = middle_curve[i+1] - middle_curve[i]

            # Sign indicates direction: positive = left turn, negative = right turn
            signed_curvatures = np.zeros(len(middle_curve)-2)
            for i in range(len(middle_curve)-2):

                v1 = vectors[i]
                v2 = vectors[i+1]
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                
                v1_mag = np.linalg.norm(v1)
                v2_mag = np.linalg.norm(v2)
                if v1_mag > 0.001 and v2_mag > 0.001:  # Avoid division by zero
                    signed_curvatures[i] = cross_z / (v1_mag * v2_mag)
                else:
                    signed_curvatures[i] = 0
            abs_curvatures = np.abs(signed_curvatures)
            
            max_curvature_idx = np.argmax(abs_curvatures)
            
            lookahead_idx = min(max_curvature_idx + self.lookahead_distance, len(abs_curvatures) - 1)
            future_curvature_region = abs_curvatures[max_curvature_idx:lookahead_idx+1]
            
            if len(future_curvature_region) > 0:
                future_max_idx = np.argmax(future_curvature_region) + max_curvature_idx
                # Blend current and future curvature points based on weights
                if abs_curvatures[future_max_idx] > abs_curvatures[max_curvature_idx] * 0.7:
                    max_curvature_idx = int(max_curvature_idx * (1 - self.future_weight) + 
                                           future_max_idx * self.future_weight)
                    
            # Skip apex calculation if curvature is too low (we're in a straight section)
            if abs_curvatures[max_curvature_idx] < self.curvature_threshold * 10:
                final_path = np.vstack([self.car_position, middle_points])
                curvature = max(self.calculate_curvature(final_path), self.min_curvature_value)               
                if self.last_path is not None and len(self.last_path) == len(final_path):
                    final_path = self.last_path * self.path_smoothing_factor + final_path * (1 - self.path_smoothing_factor)
                    curvature = self.last_curvature * self.path_smoothing_factor + curvature * (1 - self.path_smoothing_factor)
                
                self.last_path = final_path
                self.last_curvature = curvature
                return final_path, curvature
            # Use a region around the maximum curvature point to determine turn direction
            region_start = max(0, max_curvature_idx - 5)
            region_end = min(len(signed_curvatures), max_curvature_idx + 5)
            turn_region_curvatures = signed_curvatures[region_start:region_end]
            # Average sign in the region of highest curvature gives overall turn direction
            is_right_turn = np.mean(turn_region_curvatures) > 0
            inner_points = right_points if is_right_turn else left_points
            dx = np.gradient(inner_points[:, 0])
            dy = np.gradient(inner_points[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvatures = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
            apex_index = np.argmax(np.abs(curvatures))
            apex_point = inner_points[apex_index - 5]  # get early index of the curve
            # Start point
            car_forward_vector = np.array([0, -1])
            
            # Reduced back offset for faster response to curves
            back_offset = 0.5  # reduced from 1.0
            start_behind = self.car_position + car_forward_vector * back_offset
            # If we detect a curve, bias the starting point slightly in that direction
            outward_vec = np.zeros(2)
            if is_right_turn:
                outward_vec = left_curve[0] - right_curve[0]
            else:
                outward_vec = right_curve[0] - left_curve[0]
            # Normalize and apply outward offset with curve-aware adjustment
            outward_vec /= (np.linalg.norm(outward_vec) + 1e-6)
            
            curve_bias = min(1.8, abs_curvatures[max_curvature_idx] * 70) # responsiveness
            start_point = start_behind + outward_vec * curve_bias
            end_point = (left_points[-1] + right_points[-1]) / 2
            car_to_apex = apex_point - start_point
            is_apex_behind = np.dot(car_to_apex, car_forward_vector) < 0
            apex_x = apex_point[0]
            end_x = end_point[0]
            outer_behind_apex = (apex_x < end_x) if is_right_turn else (apex_x > end_x)

            if is_apex_behind:
                key_points = np.array([start_point, end_point])
            elif outer_behind_apex:
                extended_inner = (left_curve if is_right_turn else right_curve)
                extended_point = extended_inner[-1]   
                key_points = np.array([start_point, apex_point, extended_point])
            elif max_curvature_idx < self.min_curve_idx and len(abs_curvatures) > self.min_curve_idx:
                key_points = np.array([start_point, end_point])
            else:
                mid_point1 = start_point * 0.8 + apex_point * 0.2
                mid_point2 = start_point * 0.4 + apex_point * 0.6
                key_points = np.array([start_point, mid_point1, mid_point2, apex_point, end_point])

            t = np.linspace(0, 1, len(key_points))
            spline_x = CubicSpline(t, key_points[:, 0])
            spline_y = CubicSpline(t, key_points[:, 1])

            t_values = np.linspace(0, 1, self.number_of_points)
            spline_path = np.zeros((self.number_of_points, 2))
            for i, t_val in enumerate(t_values):
                spline_path[i, 0] = spline_x(t_val)
                spline_path[i, 1] = spline_y(t_val)
            curvature = self.calculate_curvature(spline_path) 
            key_points_markers = np.vstack([
                self.create_highlight_circle(start_point, radius=2, num_points=8),
                self.create_highlight_circle(apex_point, radius=3, num_points=10),
                self.create_highlight_circle(end_point, radius=2, num_points=8),
            ])
            full_path = np.vstack([spline_path, key_points_markers])
        final_path = full_path

        self.last_path = final_path
        self.last_curvature = curvature
        return final_path, curvature
    
    def fix_path_data(self, x_middle, y_middle):
        """
        Fix common issues in path data that might cause splprep to fail:
        1. Remove duplicate points
        2. Fix non-monotonic sections
        """
        # Convert to numpy arrays if not already
        x = np.array(x_middle)
        y = np.array(y_middle)
        
        # Create a mask for unique points (remove duplicates)
        unique_mask = np.ones(len(x), dtype=bool)
        
        for i in range(len(x)-1):
            # Check if current point is identical to next point
            if np.isclose(x[i], x[i+1]) and np.isclose(y[i], y[i+1]):
                unique_mask[i+1] = False
        
        # Apply the mask to keep only unique points
        x_unique = x[unique_mask]
        y_unique = y[unique_mask]
        
        # Print info about removed duplicates
        num_removed = len(x) - len(x_unique)
        if num_removed > 0:
            print(f"Removed {num_removed} duplicate points")
        
        # Option 1: Sort by arc length (not suitable for all paths)
        # This works for simple paths but may fail for complex shapes
        
        # Option 2: Calculate a custom parameterization
        # This is a more robust approach
        t = np.zeros(len(x_unique))
        t[1:] = np.sqrt(np.diff(x_unique)**2 + np.diff(y_unique)**2)
        t = np.cumsum(t)
        t /= t[-1]  # normalize to [0, 1]
        
        return x_unique, y_unique, t

    def create_highlight_circle(self, center_point, radius=5, num_points=12):
        #Creates a circle of points around a specified center point for visualization purposes.
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        circle_points = np.zeros((num_points, 2))
        for i, angle in enumerate(angles):
            x = center_point[0] + radius * np.cos(angle)
            y = center_point[1] + radius * np.sin(angle)
            circle_points[i] = [x, y]
        
        return circle_points
    

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