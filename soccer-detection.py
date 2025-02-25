import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
from collections import defaultdict

class SoccerDetectionSystem:
    def __init__(self):
        # Initialize YOLOv8 model
        try:
            self.yolo_model = YOLO("yolov8x.pt")
        except Exception as e:
            print("Downloading YOLOv8x model...")
            # This will automatically download the model if it doesn't exist
            self.yolo_model = YOLO("yolov8x")  
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize player tracking
        self.player_trackers = defaultdict(dict)
        self.player_team_mapping = {}  # Map player IDs to teams
        self.frame_count = 0
        
        # Field line detector
        self.field_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
        # Last known ball position
        self.ball_position = None
        self.ball_history = []
        
        # Team colors for visualization
        self.team_colors = {
            "team_a": (0, 0, 255),    # Red
            "team_b": (255, 0, 0),    # Blue
            "referee": (0, 255, 255), # Yellow
            "unknown": (255, 255, 255) # White
        }
    
    def detect_teams(self, frame):
        """Detect and classify players by team based on jersey colors"""
        # Convert to HSV for better color segmentation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # This is a simplified approach - in reality you'd use a more robust
        # clustering algorithm on detected player regions
        # Define team color ranges in HSV
        team_a_lower = np.array([0, 100, 100])  # Red team
        team_a_upper = np.array([10, 255, 255])
        
        team_b_lower = np.array([100, 100, 100])  # Blue team
        team_b_upper = np.array([130, 255, 255])
        
        # Create masks for each team
        team_a_mask = cv2.inRange(hsv_frame, team_a_lower, team_a_upper)
        team_b_mask = cv2.inRange(hsv_frame, team_b_lower, team_b_upper)
        
        return team_a_mask, team_b_mask
    
    def detect_field_lines(self, frame):
        """Detect soccer field lines using LSD detector"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to highlight white lines
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Detect lines
        lines = self.field_detector.detect(thresh)
        
        return lines[0] if lines[0] is not None else []
    
    def detect_ball(self, frame, detections):
        """Extract ball position from YOLO detections"""
        for detection in detections:
            if detection.boxes is not None:
                for i, box in enumerate(detection.boxes):
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Assuming class 32 is 'sports ball' in COCO dataset
                    if cls == 32 and conf > 0.6:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        self.ball_position = (int(center_x), int(center_y))
                        self.ball_history.append(self.ball_position)
                        
                        # Keep only the last 30 positions
                        if len(self.ball_history) > 30:
                            self.ball_history.pop(0)
                        
                        return True
        
        # If ball not detected in this frame but we have history, use last known position
        return False
    
    def estimate_player_velocity(self, player_id):
        """Calculate player velocity based on position history"""
        positions = self.player_trackers[player_id].get('positions', [])
        if len(positions) < 5:
            return 0, 0
        
        # Calculate velocity from last 5 positions
        recent_positions = positions[-5:]
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]
        
        # Time is represented by frames, assuming constant frame rate
        dt = 5  # 5 frames
        
        vx = dx / dt
        vy = dy / dt
        
        return vx, vy
    
    def detect_rapid_deceleration(self, player_id):
        """Detect if a player has rapidly decelerated (potential foul indicator)"""
        velocities = self.player_trackers[player_id].get('velocities', [])
        if len(velocities) < 3:
            return False
        
        # Calculate the magnitude of velocity change
        v_prev = math.sqrt(velocities[-2][0]**2 + velocities[-2][1]**2)
        v_curr = math.sqrt(velocities[-1][0]**2 + velocities[-1][1]**2)
        
        # If velocity decreased significantly
        if v_prev > 2 and v_curr < 0.5 * v_prev:
            return True
        
        return False
    
    def detect_player_collision(self, player_id):
        """Detect if a player has collided with another player"""
        player_pos = self.player_trackers[player_id].get('current_position')
        if player_pos is None:
            return False, None
        
        # Check distance to all other players
        for other_id, other_data in self.player_trackers.items():
            if other_id == player_id:
                continue
                
            other_pos = other_data.get('current_position')
            if other_pos is None:
                continue
            
            # Calculate distance between players
            distance = math.sqrt((player_pos[0] - other_pos[0])**2 + 
                               (player_pos[1] - other_pos[1])**2)
            
            # If players are very close, consider it a potential collision
            if distance < 50:  # Threshold in pixels
                return True, other_id
        
        return False, None
    
    def detect_offside(self, frame, frame_width):
        """
        Detect offside situations
        
        Simplified approach:
        1. Identify the second-to-last defender of the defending team
        2. Check if any attacking player is ahead of this defender and behind the ball
           when the ball is passed
        """
        # Get all players from each team
        team_a_players = [pid for pid, team in self.player_team_mapping.items() if team == "team_a"]
        team_b_players = [pid for pid, team in self.player_team_mapping.items() if team == "team_b"]
        
        # Determine attacking direction (simplified - in reality would need to track ball movement)
        # Assume team_a attacks from left to right
        attacking_team = team_a_players
        defending_team = team_b_players
        
        # If no players detected for either team, can't determine offside
        if not attacking_team or not defending_team:
            return False, None
        
        # Sort defending players by x-coordinate (from left to right)
        sorted_defenders = sorted(
            [self.player_trackers[pid].get('current_position', (0, 0)) for pid in defending_team],
            key=lambda pos: pos[0]
        )
        
        # Get the second last defender position (second from the right)
        if len(sorted_defenders) >= 2:
            second_last_defender_x = sorted_defenders[-2][0]
        else:
            # If only one defender, use that one
            second_last_defender_x = sorted_defenders[-1][0]
        
        # Check if any attacking player is beyond the second last defender
        offside_players = []
        for pid in attacking_team:
            player_pos = self.player_trackers[pid].get('current_position')
            if player_pos is None:
                continue
                
            # If attacker is ahead of second last defender
            if player_pos[0] > second_last_defender_x:
                # Check if the ball was just passed (simplified detection)
                if self.is_ball_passed():
                    offside_players.append(pid)
        
        if offside_players:
            return True, offside_players
        
        return False, None
    
    def is_ball_passed(self):
        """Detect if ball was just passed based on ball movement"""
        # Simple detection based on ball velocity
        if len(self.ball_history) < 10:
            return False
            
        # Calculate ball displacement
        start_pos = self.ball_history[-10]
        end_pos = self.ball_history[-1]
        
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # If ball moved significantly, consider it a pass
        return distance > 50  # Threshold in pixels
    
    def detect_foul(self, frame):
        """
        Detect potential fouls based on:
        1. Player collisions
        2. Rapid player decelerations
        3. Pose analysis for tackles
        """
        potential_fouls = []
        
        # Check each player
        for player_id in self.player_trackers:
            # Skip if no position data
            if 'current_position' not in self.player_trackers[player_id]:
                continue
                
            # 1. Check for collisions
            collision, other_id = self.detect_player_collision(player_id)
            
            # 2. Check for rapid decelerations
            rapid_decel = self.detect_rapid_deceleration(player_id)
            
            # 3. Check player's pose for potential tackling motion
            tackle_detected = self.detect_tackling_pose(player_id)
            
            # Combine signals
            if (collision and (rapid_decel or tackle_detected)):
                # Get team information
                team1 = self.player_team_mapping.get(player_id, "unknown")
                team2 = self.player_team_mapping.get(other_id, "unknown") if other_id else "unknown"
                
                # Only count as foul if different teams involved
                if team1 != team2 and team1 != "referee" and team2 != "referee":
                    foul_location = self.player_trackers[player_id]['current_position']
                    potential_fouls.append((player_id, other_id, foul_location))
        
        return potential_fouls
    
    def detect_tackling_pose(self, player_id):
        """Detect if a player's pose indicates a tackle"""
        pose_data = self.player_trackers[player_id].get('pose_landmarks')
        if pose_data is None:
            return False
            
        # Extract relevant keypoints (simplified)
        # In a real implementation, would need more sophisticated pose analysis
        try:
            # Get knee positions
            left_knee = pose_data[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_data[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            
            # Get ankle positions
            left_ankle = pose_data[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = pose_data[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Check if one leg is extended (potential tackle)
            left_leg_extended = abs(left_knee.y - left_ankle.y) > 100
            right_leg_extended = abs(right_knee.y - right_ankle.y) > 100
            
            return left_leg_extended or right_leg_extended
        except:
            return False
    
    def process_frame(self, frame):
        """Process a single frame of video"""
        self.frame_count += 1
        height, width, _ = frame.shape
        
        # 1. Run YOLOv8 detection
        detections = self.yolo_model(frame)
        
        # 2. Detect and classify players by team
        team_a_mask, team_b_mask = self.detect_teams(frame)
        
        # 3. Track ball
        ball_detected = self.detect_ball(frame, detections)
        
        # 4. Process YOLO detections to track players
        detected_player_boxes = []
        
        for detection in detections:
            if detection.boxes is not None:
                for i, box in enumerate(detection.boxes):
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Class 0 is 'person' in COCO dataset
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Create crop of player for pose estimation
                        player_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if player_crop.size == 0:
                            continue
                            
                        # Assign player to a team based on jersey color
                        player_region = np.zeros_like(frame)
                        player_region[int(y1):int(y2), int(x1):int(x2)] = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        team_a_overlap = cv2.countNonZero(cv2.bitwise_and(team_a_mask, cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)))
                        team_b_overlap = cv2.countNonZero(cv2.bitwise_and(team_b_mask, cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)))
                        
                        # Simple team assignment based on color overlap
                        team = "unknown"
                        if team_a_overlap > team_b_overlap and team_a_overlap > 50:
                            team = "team_a"
                        elif team_b_overlap > team_a_overlap and team_b_overlap > 50:
                            team = "team_b"
                        
                        # Generate a unique ID based on initial position (simplified)
                        # In practice, would use a more robust tracking algorithm
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        player_id = f"player_{int(center_x)}_{int(center_y)}"
                        
                        # Find closest tracked player to associate with
                        min_dist = float('inf')
                        closest_id = None
                        
                        for tracked_id, data in self.player_trackers.items():
                            if 'current_position' in data:
                                tx, ty = data['current_position']
                                dist = math.sqrt((center_x - tx)**2 + (center_y - ty)**2)
                                if dist < min_dist and dist < 50:  # Threshold for association
                                    min_dist = dist
                                    closest_id = tracked_id
                        
                        # Use existing ID if found, otherwise create new
                        if closest_id:
                            player_id = closest_id
                        
                        # Store player position
                        self.player_trackers[player_id]['current_position'] = (int(center_x), int(center_y))
                        
                        # Update position history
                        if 'positions' not in self.player_trackers[player_id]:
                            self.player_trackers[player_id]['positions'] = []
                        self.player_trackers[player_id]['positions'].append((int(center_x), int(center_y)))
                        
                        # Keep only recent positions
                        if len(self.player_trackers[player_id]['positions']) > 30:
                            self.player_trackers[player_id]['positions'].pop(0)
                        
                        # Calculate velocity
                        vx, vy = self.estimate_player_velocity(player_id)
                        if 'velocities' not in self.player_trackers[player_id]:
                            self.player_trackers[player_id]['velocities'] = []
                        self.player_trackers[player_id]['velocities'].append((vx, vy))
                        
                        # Keep only recent velocities
                        if len(self.player_trackers[player_id]['velocities']) > 10:
                            self.player_trackers[player_id]['velocities'].pop(0)
                        
                        # Update team mapping
                        if team != "unknown":
                            self.player_team_mapping[player_id] = team
                        
                        # Store the detection box
                        detected_player_boxes.append({
                            'id': player_id,
                            'box': (int(x1), int(y1), int(x2), int(y2)),
                            'team': team
                        })
                        
                        # Pose estimation on player crop (for advanced foul detection)
                        try:
                            # Resize player crop to consistent dimensions before pose detection
                            POSE_INPUT_SIZE = (256, 256)  # Choose dimensions that work for your use case
                            player_crop_resized = cv2.resize(player_crop, POSE_INPUT_SIZE)
                            
                            results = self.pose.process(cv2.cvtColor(player_crop_resized, cv2.COLOR_BGR2RGB))
                            if results.pose_landmarks:
                                # Scale landmarks back to original crop size
                                scale_x = player_crop.shape[1] / POSE_INPUT_SIZE[0]
                                scale_y = player_crop.shape[0] / POSE_INPUT_SIZE[1]
                                
                                scaled_landmarks = []
                                for landmark in results.pose_landmarks.landmark:
                                    scaled_landmark = type(landmark)()
                                    scaled_landmark.x = landmark.x * scale_x
                                    scaled_landmark.y = landmark.y * scale_y
                                    scaled_landmark.z = landmark.z  # Z coordinate can stay the same
                                    scaled_landmark.visibility = landmark.visibility
                                    scaled_landmarks.append(scaled_landmark)
                                
                                self.player_trackers[player_id]['pose_landmarks'] = scaled_landmarks
                        except Exception as e:
                            # Gracefully handle any pose detection failures
                            print(f"Pose detection failed for player {player_id}: {str(e)}")
                            pass
        
        # 5. Detect field lines
        field_lines = self.detect_field_lines(frame)
        
        # 6. Detect offsides
        is_offside, offside_players = self.detect_offside(frame, width)
        
        # 7. Detect fouls
        potential_fouls = self.detect_foul(frame)
        
        # 8. Visualize results
        result_frame = self.visualize_results(
            frame.copy(), 
            detected_player_boxes, 
            field_lines, 
            is_offside, 
            offside_players,
            potential_fouls
        )
        
        return result_frame, is_offside, potential_fouls
    
    def visualize_results(self, frame, player_boxes, field_lines, is_offside, offside_players, potential_fouls):
        """Visualize detection results on frame"""
        # Draw field lines
        if field_lines is not None and len(field_lines) > 0:
            for line in field_lines:
                try:
                    # Handle different possible line formats
                    if isinstance(line, np.ndarray):
                        x1, y1, x2, y2 = line.flatten()[:4]
                    else:
                        x1, y1, x2, y2 = line[:4]
                    
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                except Exception as e:
                    print(f"Failed to draw line: {str(e)}")
                    continue
        
        # Draw players
        for player in player_boxes:
            x1, y1, x2, y2 = player['box']
            team = player['team']
            player_id = player['id']
            
            # Get team color
            color = self.team_colors.get(team, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID and team
            cv2.putText(frame, f"{player_id[:3]}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw position history (trail)
            positions = self.player_trackers[player_id].get('positions', [])
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i-1], positions[i], color, 1)
        
        # Draw ball position and history
        if self.ball_position:
            cv2.circle(frame, self.ball_position, 5, (0, 165, 255), -1)
            
            # Draw ball trail
            for i in range(1, len(self.ball_history)):
                cv2.line(frame, self.ball_history[i-1], self.ball_history[i], (0, 165, 255), 2)
        
        # Highlight offside
        if is_offside and offside_players:
            # Draw offside line
            team_b_players = [pid for pid, team in self.player_team_mapping.items() if team == "team_b"]
            sorted_defenders = sorted(
                [self.player_trackers[pid].get('current_position', (0, 0)) for pid in team_b_players],
                key=lambda pos: pos[0]
            )
            
            if len(sorted_defenders) >= 2:
                offside_x = sorted_defenders[-2][0]
                cv2.line(frame, (int(offside_x), 0), (int(offside_x), frame.shape[0]), (0, 0, 255), 2)
                
                # Add offside text
                cv2.putText(frame, "OFFSIDE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Highlight offside players
                for player_id in offside_players:
                    if player_id in self.player_trackers and 'current_position' in self.player_trackers[player_id]:
                        pos = self.player_trackers[player_id]['current_position']
                        cv2.circle(frame, pos, 30, (0, 0, 255), 3)
        
        # Highlight fouls
        for player1_id, player2_id, foul_location in potential_fouls:
            # Draw foul location
            cv2.circle(frame, foul_location, 40, (0, 0, 255), 3)
            
            # Add foul text
            cv2.putText(frame, "FOUL DETECTED", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Draw line between players involved in foul
            if player2_id and player2_id in self.player_trackers and 'current_position' in self.player_trackers[player2_id]:
                player1_pos = self.player_trackers[player1_id]['current_position']
                player2_pos = self.player_trackers[player2_id]['current_position']
                cv2.line(frame, player1_pos, player2_pos, (0, 0, 255), 2)
        
        return frame

def process_video(video_path, output_path=None):
    """Process a soccer video to detect fouls and offsides"""
    # Initialize detector
    detector = SoccerDetectionSystem()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if output path provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Lists to store detections
    offside_frames = []
    foul_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result_frame, is_offside, potential_fouls = detector.process_frame(frame)
        
        # Store frames with violations
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if is_offside:
            offside_frames.append(frame_num)
        
        if potential_fouls:
            foul_frames.append(frame_num)
        
        # Write to output video
        if output_path:
            out.write(result_frame)
        
        # Display frame (optional, for debugging)
        cv2.imshow('Soccer Analysis', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Return violations
    return offside_frames, foul_frames

def save_violation_frames(video_path, frame_numbers, output_dir):
    """Save specific frames from video to output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    
    for frame_num in frame_numbers:
        # Set video position to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Save frame
        output_path = os.path.join(output_dir, f"violation_frame_{frame_num}.jpg")
        cv2.imwrite(output_path, frame)
    
    cap.release()

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Soccer Foul and Offside Detection")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file")
    parser.add_argument("--save-frames", action="store_true", help="Save frames with violations")
    parser.add_argument("--output-dir", default="violation_frames", help="Directory to save violation frames")
    
    args = parser.parse_args()
    
    # Process video
    offside_frames, foul_frames = process_video(args.video, args.output)
    
    # Print results
    print(f"Detected {len(offside_frames)} offside situations at frames: {offside_frames}")
    print(f"Detected {len(foul_frames)} potential fouls at frames: {foul_frames}")
    
    # Save violation frames if requested
    if args.save_frames:
        # Save offside frames
        offside_dir = os.path.join(args.output_dir, "offside")
        save_violation_frames(args.video, offside_frames, offside_dir)
        
        # Save foul frames
        foul_dir = os.path.join(args.output_dir, "fouls")
        save_violation_frames(args.video, foul_frames, foul_dir)
        
        print(f"Saved violation frames to {args.output_dir}")
