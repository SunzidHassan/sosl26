from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
import cv2
import os
import pandas as pd
import numpy as np

from sOSL_utils import world_to_grid

# ==========================
# Vision Functions
# ==========================

def boxDepth(x, y, w, h, controller):
    """Estimates the depth of an object based on its bounding box in the depth frame.

    Calculates the 90th percentile depth value within the bounding box region
    defined by (x, y, w, h) in the controller's last depth frame.

    Parameters
    ----------
    x : int
        Center x-coordinate of the bounding box (in pixels).
    y : int
        Center y-coordinate of the bounding box (in pixels).
    w : int
        Width of the bounding box (in pixels).
    h : int
        Height of the bounding box (in pixels).
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.

    Returns
    -------
    float
        The estimated depth, rounded to one decimal place. Returns 0 or NaN if
        the box is invalid or depth data is missing.
    """
    # Calculate pixel bounds, ensuring they are within frame dimensions
    frame_h, frame_w = controller.last_event.depth_frame.shape[:2]
    vMin = max(0, y - h // 2)
    vMax = min(frame_h, y + h // 2)
    hMin = max(0, x - w // 2)
    hMax = min(frame_w, x + w // 2)

    # Check if the box has valid dimensions
    if vMin >= vMax or hMin >= hMax:
        print(f"Warning: Invalid bounding box dimensions [{vMin}:{vMax}, {hMin}:{hMax}] for depth calculation.")
        return 0.0 # Or np.nan

    depthFrame = controller.last_event.depth_frame
    # Extract depth values within the box
    depth_values = depthFrame[vMin:vMax, hMin:hMax]

    # Check if depth_values is empty (e.g., due to invalid box slicing)
    if depth_values.size == 0:
         print(f"Warning: No depth values found in box [{vMin}:{vMax}, {hMin}:{hMax}].")
         return 0.0 # Or np.nan

    # Calculate 90th percentile
    boxDepth = np.percentile(depth_values, 90)
    return round(boxDepth, 1)


def coord23D_focal(x, y, w, h, controller):
    """
    Converts 2D pixel coordinates to 3D world coordinates using the 
    Focal Length (Intrinsic) method.
    """
    # --- 1. Camera Intrinsics ---
    W, H = 300, 300
    fov = 90
    # Calculate focal length: f = W / (2 * tan(FOV/2))
    f = W / (2 * np.tan(np.deg2rad(fov / 2)))
    
    # Principal point (center of the image)
    cx, cy = W / 2.0, H / 2.0

    # --- 2. Estimate Depth ---
    # Assuming boxDepth is defined elsewhere as in your previous snippet
    d = boxDepth(x, y, w, h, controller)
    if d <= 0:
        return 0.0, 0.0, 0.0

    # --- 3. Back-projection to Camera Space ---
    # In AI2-THOR (Unity), Camera Space is: +X Right, +Y Up, +Z Forward
    # (u, v) pixels: u=0 is left, v=0 is top.
    X_c = (x - cx) * d / f
    Y_c = -(y - cy) * d / f  # Negative because pixel 'y' increases downwards
    Z_c = d

    # --- 4. Coordinate Transformation to World Space ---
    event = controller.last_event
    
    # Get Camera World Position (more accurate than Agent Position)
    cam_pos = event.metadata['cameraPosition']
    tx, ty, tz = cam_pos['x'], cam_pos['y'], cam_pos['z']

    # Get Rotation Angles (converted to Radians)
    # Yaw: Agent's rotation around Y axis
    # Pitch: Camera's horizon (rotation around X axis)
    yaw = np.deg2rad(event.metadata['agent']['rotation']['y'])
    pitch = np.deg2rad(event.metadata['agent']['cameraHorizon'])

    # Pitch Matrix (Rotation around X)
    # In AI2-THOR, positive pitch is looking DOWN.
    # R_pitch = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(pitch), -np.sin(pitch)],
    #     [0, np.sin(pitch), np.cos(pitch)]
    # ])

    # Yaw Matrix (Rotation around Y)
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Combine: Local -> Pitched -> Yawed
    P_camera = np.array([X_c, Y_c, Z_c])
    P_world_rotated = R_yaw @ P_camera

    # Add translation to get Global Coordinates
    final_x = P_world_rotated[0] + tx
    final_y = P_world_rotated[1] + ty
    final_z = P_world_rotated[2] + tz

    return round(final_x, 3), round(final_y, 3), round(final_z, 3)

def visionBranch(model, itemDF, controller, save_dir, step_count, fusionMode = None, confThr=0.1):
    """Detects objects using YOLO, estimates their 3D position, and updates a DataFrame.

    Runs YOLOv8 on the current camera frame. For each detection above `confThr`:
    1. Estimates the 3D world coordinates using `coord23D`.
    2. Checks if an object of the same class already exists in `itemDF` nearby (dist < 0.5).
    3. If nearby object exists, averages its position with the new detection.
    4. If no nearby object exists, adds the new detection as a new row in `itemDF`.

    Parameters
    ----------
    model : ultralytics.YOLO
        The loaded YOLOv8 model instance.
    itemDF : pd.DataFrame
        DataFrame containing information about previously detected objects.
        Expected columns: 'objectType' (str), 'Position' (str "x, y, z"), 'Conf' (float).
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.
    confThr : float, optional
        Confidence threshold for YOLO detections. Defaults to 0.3.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with new or averaged object detections.
    """
    # Get current frame and run YOLO detection
    current_frame = np.array(controller.last_event.frame)

    object_metadata = controller.last_event.metadata["objects"]

    # List what you DON'T want
    exclude_names = ['Cabinet', 'Cabinet_opened', 'CounterTop', 'Drawer', 'Drawer_opened', 'Floor', 'Shelf', 'Window', 'Apple_sliced', 'Bowl_filled']

    # Create list of IDs for everything else
    # target_classes = [idx for idx, name in model.names.items() if name not in exclude_names]
    target_names = ['Apple', 'Book', 'Bottle', 'Bowl', 'Bread', 'ButterKnife', 'Cabinet', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'DishSponge', 'Drawer', 'Egg', 'Faucet', 'Fork', 'Fridge', 'GarbageCan', 'HousePlant', 'Kettle', 'Knife', 'Lettuce', 'LightSwitch', 'Microwave', 'Mug', 'Pan', 'PaperTowelRoll', 'PepperShaker', 'Plate', 'Pot', 'Potato', 'SaltShaker', 'Shelf', 'ShelvingUnit', 'Sink', 'SoapBottle', 'Spatula', 'Spoon', 'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'Toaster', 'Tomato', 'Vase', 'WineBottle']
    target_classes = [idx for idx, name in model.names.items() if name in target_names]
    # Run inference
    results = model(np.array(controller.last_event.frame), classes=target_classes)

    # show yolo detections in a window
    # results[0].plot()

    # Make a copy to avoid modifying the original DataFrame passed in
    updated_itemDF = itemDF.copy()

    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{save_dir}/yolo_step{step_count}.jpg", annotated_img_rgb)
    cv2.imshow("YOLO Result", annotated_img_rgb)
    cv2.waitKey(1) # Display the window briefly; adjust as needed for your environment

    # Process detections
    for box in results[0].boxes:
        confidence = box.conf[0].item()
        if confidence > confThr:
            class_id = int(box.cls[0].item())
            className = model.names[class_id]
            
            if className in exclude_names:
                continue # Skip excluded classes
            # Extract detection info
            # print(f"Detected {className}")
            x, y, w, h = box.xywh[0] # Center x, y, width, height

            x_pix, y_pix, w_pix, h_pix = round(x.item()), round(y.item()), round(w.item()), round(h.item())

            # Estimate 3D position
            if fusionMode == 'focal':
                ## Focal length method
                x_glob, y_glob, z_glob = coord23D_focal(x_pix, y_pix, w_pix, h_pix, controller)

            elif fusionMode == 'GT':
                ## Ground truth lookup
                object_info = next((obj for obj in object_metadata if obj["objectType"] == className), None)
                # print(f"Object metadata for {className}: {object_info}")
                x_glob, y_glob, z_glob = object_info['position']['x'], object_info['position']['y'], object_info['position']['z']
            else:
                print(f"Warning: Unknown fusionMode '{fusionMode}' specified. Defaulting to focal method.")
                x_glob, y_glob, z_glob = coord23D_focal(x_pix, y_pix, w_pix, h_pix, controller)


            # Skip if coord23D failed
            if x_glob == 0.0 and y_glob == 0.0 and z_glob == 0.0:
                 continue
            new_position = np.array([x_glob, y_glob, z_glob])

            updated = False
            # --- *** CHECK IF DATAFRAME IS EMPTY *** ---
            # Only try to match if the DataFrame has data and the necessary column
            if not updated_itemDF.empty and 'objectType' in updated_itemDF.columns:
                match_indices = updated_itemDF.index[updated_itemDF['objectType'] == className].tolist()

                for idx in match_indices:
                    try:
                        # Parse existing position string
                        existing_position_str = updated_itemDF.loc[idx, 'Position']
                        existing_position = np.array([float(val.strip()) for val in existing_position_str.split(',')])

                        # Check distance
                        dist = np.linalg.norm(new_position - existing_position)
                        if dist < 0.1: # don't take average: 0.1, take average: 10.0
                            # Average positions if close enough
                            avg_position = (new_position + existing_position) / 2.0
                            updated_itemDF.loc[idx, 'Position'] = f"{avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f}"
                            # Optionally update confidence
                            updated_itemDF.loc[idx, 'Conf'] = max(confidence, updated_itemDF.loc[idx, 'Conf'])
                            updated = True
                            # break # Stop checking once updated # TODO
                    except Exception as e:
                        print(f"Error processing existing position for {className} at index {idx}: {e}")
                        continue # Skip this entry if parsing fails
            # --- *** END CHECK *** ---

            # If no nearby existing object was found/updated, add as new row
            if not updated:
                new_row_data = {
                    "objectType": [className],
                    "Conf": [confidence],
                    "Position": [f"{x_glob:.2f}, {y_glob:.2f}, {z_glob:.2f}"]
                }
                new_row_df = pd.DataFrame(new_row_data)

                # Use concat, ensuring columns align even if updated_itemDF was initially empty
                updated_itemDF = pd.concat([updated_itemDF, new_row_df], ignore_index=True)


    # Ensure essential columns exist before returning, even if no objects detected
    # This prevents errors later if no objects are found in the initial scan
    for col in ['objectType', 'Conf', 'Position']:
         if col not in updated_itemDF.columns:
              updated_itemDF[col] = pd.Series(dtype='object' if col != 'Conf' else 'float')

    return updated_itemDF


def initialize_envKnowledge(controller, model, itemDF, save_path, confThr=0.3, fusionMode=None):
    """Initializes the environment knowledge by scanning the surroundings.

    Rotates the agent 360 degrees (4 steps of 90 degrees), calling `visionBranch`
    at each step to populate the `itemDF` DataFrame with detected objects.

    Parameters
    ----------
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.
    model : ultralytics.YOLO
        The loaded YOLOv8 model instance.
    itemDF : pd.DataFrame
        An empty DataFrame to be populated with initial object detections.
    probMap : np.ndarray
        The current Bayesian probability map (unused in this function directly,
        but might be intended for later use or passed down).
    x_points : np.ndarray
        1D array of x-coordinates defining the grid columns (unused).
    z_points : np.ndarray
        1D array of z-coordinates defining the grid rows (unused).
    confThr : float, optional
        Confidence threshold for YOLO detections passed to `visionBranch`. Defaults to 0.3.

    Returns
    -------
    pd.DataFrame
        The `itemDF` DataFrame populated with objects detected during the scan.
    """
    current_itemDF = itemDF.copy() # Start with the (presumably empty) DataFrame
    num_rotations = 4 # 360 degrees / 90 degrees per step

    print("Initializing environment knowledge by rotating...")
    for i in range(num_rotations):
        print(f"Rotation step {i+1}/{num_rotations}")
        # Forward fusionMode so visionBranch uses the requested depth method
        current_itemDF = visionBranch(model, current_itemDF, controller, save_path, i+100, fusionMode=fusionMode, confThr=confThr)
        # Convert to dict and back to handle potential duplicate index issues if concat runs oddly
        itemDF_list = current_itemDF.to_dict(orient='records')
        current_itemDF = pd.DataFrame(itemDF_list)
        print(f"Detected items after step {i+1}:")
        print(current_itemDF.head())
        print("---")

        try:
            intialFrame = controller.last_event.cv2img  # AI2-THOR gives frame in BGR
            frame_filename = os.path.join(save_path, f"initializationFrame_{i}.png")
            cv2.imwrite(frame_filename, intialFrame)
        except Exception as e:
            print(f"Warning: Could not save initialization frame {i} to {save_path}: {e}")

        # Rotate for the next view, unless it's the last step
        if i < num_rotations - 1:
            controller.step("RotateLeft", degrees=90) # Use explicit degrees

    print("Finished initialization scan.")
    # The add_goal_similarity call was commented out, keep it that way unless needed here.
    return current_itemDF


def add_goal_similarity(itemDF, goal_phrase, probMap, x_points, z_points, alg_choice='F'):
    """Calculates and adds multimodal similarity scores to the item DataFrame.

    For each object in `itemDF`, calculates:
    - `visionSim`: Based on detection confidence (`Conf` column).
    - `langSim`: Cosine similarity between the object name embedding and the `goal_phrase` embedding.
    - `olfactionSim`: The value from the `probMap` corresponding to the object's grid location.
    - `goalSim`: A combined score based on the `alg_choice`:
        - 'f' (fusion): langSim * olfactionSim
        - 'v' (vision): langSim
        - 'o' (olfaction): olfactionSim

    The DataFrame is then sorted by `goalSim` descending.

    Parameters
    ----------
    itemDF : pd.DataFrame
        DataFrame with detected objects. Requires columns 'objectType', 'Conf', 'Position'.
    goal_phrase : str
        The textual description of the search goal (e.g., "source of smoke odor").
    probMap : np.ndarray
        The current 2D Bayesian belief map.
    x_points : np.ndarray
        1D array of x-coordinates defining the grid columns.
    z_points : np.ndarray
        1D array of z-coordinates defining the grid rows.
    alg_choice : str, optional
        Determines how `goalSim` is calculated ('f', 'v', or 'o'). Defaults to 'f'.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with added similarity columns ('visionSim', 'langSim',
        'olfactionSim', 'goalSim') and sorted by 'goalSim' descending.
    """
    if itemDF.empty:
        print("Warning: itemDF is empty in add_goal_similarity. Returning empty DataFrame.")
        # Ensure the columns exist even if empty
        for col in ["visionSim", "olfactionSim", "langSim", "goalSim"]:
             if col not in itemDF.columns:
                  itemDF[col] = np.nan
        return itemDF

    goal_embedding = model.encode(goal_phrase, convert_to_tensor=True)

    # Initialize columns if they don't exist
    for col in ["visionSim", "olfactionSim", "langSim", "goalSim"]:
        if col not in itemDF.columns:
            itemDF[col] = np.nan

    # Calculate similarities row by row
    for idx, row in itemDF.iterrows():
        object_type = row["objectType"]

        # Vision Similarity (simply confidence)
        itemDF.loc[idx, "visionSim"] = row["Conf"]

        # Language Similarity
        object_embedding = model.encode(object_type, convert_to_tensor=True)
        # Ensure embeddings are on the same device if using GPU
        # goal_embedding = goal_embedding.to(object_embedding.device)
        lang_similarity = util.pytorch_cos_sim(object_embedding, goal_embedding).item()
        itemDF.loc[idx, "langSim"] = lang_similarity

        # Olfaction Similarity
        try:
            pos_str = row["Position"]
            x_world, _, z_world = map(float, pos_str.split(','))
            grid_indices = world_to_grid(x_world, z_world, x_points, z_points)
            grid_row, grid_col = grid_indices[0], grid_indices[1] # Extract row and column

            # Ensure indices are within bounds
            if 0 <= grid_row < probMap.shape[0] and 0 <= grid_col < probMap.shape[1]:
                olf_val = probMap[grid_row, grid_col]
                itemDF.loc[idx, "olfactionSim"] = olf_val
            else:
                 print(f"Warning: Calculated grid indices ({grid_row}, {grid_col}) for object {object_type} at ({x_world:.2f}, {z_world:.2f}) are out of probMap bounds ({probMap.shape}). Setting olfactionSim to 0.")
                 itemDF.loc[idx, "olfactionSim"] = 0.0 # Assign a default value
                 olf_val = 0.0 # Use default for combined calculation
        except Exception as e:
            print(f"Error calculating olfaction similarity for object {object_type} at index {idx}: {e}")
            itemDF.loc[idx, "olfactionSim"] = 0.0 # Assign a default value on error
            olf_val = 0.0

        # Combined Goal Similarity based on alg_choice
        # Use .loc to ensure values are properly assigned back to the DataFrame slice
        vision_sim = itemDF.loc[idx, "visionSim"]
        if alg_choice == "F" or alg_choice == "G":
            combined_sim = lang_similarity * olf_val
        elif alg_choice == "V":
            combined_sim = lang_similarity
        elif alg_choice == "O":
            combined_sim = olf_val
        else: # Default or unknown mode, maybe just use fusion?
            print(f"Warning: Unknown alg_choice '{alg_choice}'. Defaulting to fusion.")
            combined_sim = lang_similarity * olf_val
        itemDF.loc[idx, "goalSim"] = combined_sim

    # Sort by combined similarity (handle potential NaNs by placing them last)
    itemDF.sort_values(by="goalSim", ascending=False, inplace=True, na_position='last')

    # Print the location of the highest belief in the probability map
    if probMap.size > 0: # Check if probMap is not empty
        max_index = np.unravel_index(np.argmax(probMap), probMap.shape)
        # Ensure indices are within bounds before accessing x_points/z_points
        if max_index[1] < len(x_points) and max_index[0] < len(z_points):
             max_x = x_points[max_index[1]]
             max_z = z_points[max_index[0]]
             print(f"Highest olfactory belief map coordinate (grid index {max_index}): world x={max_x:.2f}, z={max_z:.2f}")
        else:
             print(f"Warning: Max belief index {max_index} is out of bounds for x_points/z_points.")
    else:
         print("Warning: probMap is empty, cannot find highest belief coordinate.")


    return itemDF