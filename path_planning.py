
import numpy as np
from scipy.interpolate import CubicSpline, make_splprep, splprep, splev
class PathPlanning:
    def __init__(self):        
        
        # Fixed Starting Point in front of the car
        self.car_position = np.array([48, 68])
        
        self.startup_mode = True
        self.startup_counter = 45  # first "frames" for startup mode so the zoom doesnt create a weird path
        
        # Stage selection (enable/disable different planning strategies)

        self.just_use_middle_path = True # Stage 1
        self.check_straight_line = True  # Stage 2 - Doesnt work on by itself but can optimize double or wide curves
        self.other_splines = True # Stage 3 - Spline creation with the Apex-point to maximise the radius

        self.number_of_points = 10 # How many points should get returend 

        # Stage 3 Parameters/Storage Variables
        self.apex_history = [] # 
        self.max_history_length = 5  # Reduced from 10 for faster response
        self.apex_safety_margin = 0 # how many pixels away the apex should be from the inner lane       
        self.curvature_threshold = 0.001  # Reduced from 0.001      
        self.predict_point_range = 20 # how much of a further point we use if outer lane is behind apex  
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

        # Stage 1: Only use Middle Points in front of the car
        if(self.just_use_middle_path):
            forward_mask = middle_points[:, 1] < self.car_position[1]
            middle_points = middle_points[forward_mask]
            curvature = self.calculate_curvature(middle_points)
            return middle_points, curvature
        

        # STAGE 2: Check if straight line is valid
        if self.check_straight_line and len(left_points) >= 2:
            straight_path = self.check_straight_line_validity(left_points, right_points)
            if straight_path is not None:
                # Add car position as first point in the path
                full_path = np.vstack([self.car_position, straight_path])
                
                # FIXED: Ensure curvature is never zero for steering control
                curvature = max(self.calculate_curvature(full_path), self.min_curvature_value)
                
                # Apply path smoothing with previous path
                if self.last_path is not None and len(self.last_path) == len(full_path):
                    full_path = self.last_path * self.path_smoothing_factor + full_path * (1 - self.path_smoothing_factor)
                    curvature = self.last_curvature * self.path_smoothing_factor + curvature * (1 - self.path_smoothing_factor)
                
                self.last_path = full_path
                self.last_curvature = curvature
                return full_path, curvature

        # Stage 3: 
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
            # Determine if it's a left or right turn
            middle_curve = (left_curve + right_curve) / 2 
            # Get vectors along the path
            vectors = np.zeros((len(middle_curve)-1, 2))
            for i in range(len(middle_curve)-1):
                vectors[i] = middle_curve[i+1] - middle_curve[i]
            # Sign indicates direction: positive = left turn, negative = right turn
            signed_curvatures = np.zeros(len(middle_curve)-2)
            for i in range(len(middle_curve)-2):
                # The z-component of cross product between consecutive vectors gives turn direction
                v1 = vectors[i]
                v2 = vectors[i+1]
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                
                # Normalize by vector magnitudes for more stable results
                v1_mag = np.linalg.norm(v1)
                v2_mag = np.linalg.norm(v2)
                if v1_mag > 0.001 and v2_mag > 0.001:  # Avoid division by zero
                    signed_curvatures[i] = cross_z / (v1_mag * v2_mag)
                else:
                    signed_curvatures[i] = 0
            # NEW: Look ahead for upcoming curves
            abs_curvatures = np.abs(signed_curvatures)
            
            # IMPROVED: Better handling of curvature detection
            max_curvature_idx = np.argmax(abs_curvatures)
            
            # NEW: Look ahead for upcoming curves within lookahead distance
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
            # Use a region around the maximum curvature point to determine turn direction
            region_start = max(0, max_curvature_idx - 5)
            region_end = min(len(signed_curvatures), max_curvature_idx + 5)
            turn_region_curvatures = signed_curvatures[region_start:region_end]
            # Average sign in the region of highest curvature gives overall turn direction
            is_right_turn = np.mean(turn_region_curvatures) > 0
            # FIXED: Better apex calculation based on curve direction
            inner_points = right_points if is_right_turn else left_points
            dx = np.gradient(inner_points[:, 0])
            dy = np.gradient(inner_points[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvatures = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
            apex_index = np.argmax(np.abs(curvatures))
            apex_point = inner_points[apex_index - 5]  # get early index of the curve
            # Start point
            # IMPROVED: Start point calculation
            car_forward_vector = np.array([0, -1])
            
            # Reduced back offset for faster response to curves
            back_offset = 0.5  # reduced from 1.0
            start_behind = self.car_position + car_forward_vector * back_offset
            # NEW: Adjust start point based on upcoming curve
            # If we detect a curve, bias the starting point slightly in that direction
            outward_vec = np.zeros(2)
            if is_right_turn:
                outward_vec = left_curve[0] - right_curve[0]
            else:
                outward_vec = right_curve[0] - left_curve[0]
            # Normalize and apply outward offset with curve-aware adjustment
            outward_vec /= (np.linalg.norm(outward_vec) + 1e-6)
            
            # IMPROVED: Adjust outward bias based on curve steepness
            curve_bias = min(1.8, abs_curvatures[max_curvature_idx] * 70)  # Increased responsiveness
            start_point = start_behind + outward_vec * curve_bias
            # End point
            end_point = (left_points[-1] + right_points[-1]) / 2
            # Check if apex is behind the car
            car_to_apex = apex_point - start_point
            is_apex_behind = np.dot(car_to_apex, car_forward_vector) < 0
            # Determine if apex is horizontally ahead of end point (in global X)
            apex_x = apex_point[0]
            end_x = end_point[0]
            # Determine if the outer end is behind the apex (mirrored spline case)
            outer_behind_apex = (apex_x < end_x) if is_right_turn else (apex_x > end_x)
            # If Apex is behind the car, we made the curve
            if is_apex_behind:
                key_points = np.array([start_point, end_point])
                        # Extend the path using a point after the apex on the inner lane if outer lane is behind apex
            elif outer_behind_apex:
                extended_inner = (left_curve if is_right_turn else right_curve)
                extended_point = extended_inner[-1]   
                key_points = np.array([start_point, apex_point, extended_point])
            elif max_curvature_idx < self.min_curve_idx and len(abs_curvatures) > self.min_curve_idx:
                key_points = np.array([start_point, end_point])
            else:
                # IMPROVED: More points for smoother entry into curves
                mid_point1 = start_point * 0.8 + apex_point * 0.2
                mid_point2 = start_point * 0.4 + apex_point * 0.6
                key_points = np.array([start_point, mid_point1, mid_point2, apex_point, end_point])
            # Create a spline through these points
            t = np.linspace(0, 1, len(key_points))
            spline_x = CubicSpline(t, key_points[:, 0])
            spline_y = CubicSpline(t, key_points[:, 1])
            # Sample the spline
            t_values = np.linspace(0, 1, self.number_of_points)
            spline_path = np.zeros((self.number_of_points, 2))
            for i, t_val in enumerate(t_values):
                spline_path[i, 0] = spline_x(t_val)
                spline_path[i, 1] = spline_y(t_val)
            # For visualization, add markers at the key points
            key_points_markers = np.vstack([
                self.create_highlight_circle(start_point, radius=2, num_points=8),
                self.create_highlight_circle(apex_point, radius=3, num_points=10),
                self.create_highlight_circle(end_point, radius=2, num_points=8),
            ])
            full_path = np.vstack([spline_path, key_points_markers])
        
        # Store path and curvature for next iteration
        self.last_path = final_path
        self.last_curvature = curvature
        return final_path, curvature
    
    def create_highlight_circle(self, center_point, radius=5, num_points=12):
        #Creates a circle of points around a specified center point for visualization purposes.
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        circle_points = np.zeros((num_points, 2))
        for i, angle in enumerate(angles):
            x = center_point[0] + radius * np.cos(angle)
            y = center_point[1] + radius * np.sin(angle)
            circle_points[i] = [x, y]
        
        return circle_points
    
    def check_straight_line_validity(self, left_points, right_points):
        if len(left_points) < 2 or len(right_points) < 2:
            return None
            
        # Get furthest visible point in middle of path
        end_point = (left_points[-1] + right_points[-1]) / 2
        
        # Create straight line path from car to end point
        t = np.linspace(0, 1, len(left_points))
        straight_path = np.outer(1-t, self.car_position) + np.outer(t, end_point)
        # cut the first 20 points to not use rest curve segments on the screen
        left_points = left_points[20:]
        right_points = right_points[20:]
        
        for i in range(len(straight_path)):
            # Create line segments between lane boundary points
            if i < len(left_points) - 1 and i < len(right_points) - 1:
                left_segment = np.array([left_points[i], left_points[i+1]])
                right_segment = np.array([right_points[i], right_points[i+1]])
                
                # Check if path point is inside the lane segment
                point = straight_path[i]
                in_lane = self.is_point_inside_lane(point, left_segment, right_segment)
                
                if not in_lane:
                    return None 
        
        return straight_path[1:]
    
    def is_point_inside_lane(self, point, left_segment, right_segment):
        # Check left boundary (point should be on right side)
        left_vec = left_segment[1] - left_segment[0]
        left_to_point = point - left_segment[0]
        left_cross = np.cross(left_vec, left_to_point)
        
        # Check right boundary (point should be on left side)
        right_vec = right_segment[1] - right_segment[0]
        right_to_point = point - right_segment[0]
        right_cross = np.cross(right_vec, right_to_point)
        
        # Point is inside lane if it's on the right of left boundary and left of right boundary
        return left_cross > 0 and right_cross < 0
    
    def calculate_middle_path(self, left_points, right_points):
        """
        Stage 2: Calculate the middle path between lane boundaries
        """
        # Calculate middle points
        middle_points = (left_points + right_points) / 2
        return middle_points
    
    def calculate_local_curvatures(self, path):
        """Calculate curvature at each point in the path"""
        if len(path) < 3:
            return np.zeros(len(path))
            
        curvatures = np.zeros(len(path))
        
        for i in range(1, len(path) - 1):
            p1 = path[i - 1]
            p2 = path[i]
            p3 = path[i + 1]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Vector lengths
            len_v1 = np.linalg.norm(v1)
            len_v2 = np.linalg.norm(v2)
            
            if len_v1 < 1e-6 or len_v2 < 1e-6:
                curvatures[i] = 0
                continue
            
            # Normalize vectors
            v1_norm = v1 / len_v1
            v2_norm = v2 / len_v2
            
            # Angle between vectors
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot_product)
            
            # Curvature is approximately angle/distance
            curvatures[i] = np.abs(angle) / ((len_v1 + len_v2) / 2)
            
        return curvatures
    
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