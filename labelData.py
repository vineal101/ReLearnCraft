import json
import numpy as np
import torch as th
import cv2
import pickle
from argparse import ArgumentParser
from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent

CAMERA_SCALER = 360.0 / 2400.0

def json_action_from_prediction(predicted_action):
    """
    Converts a predicted action to the desired JSON format.
    """
    mouse = {
        "x": 0.0,  # Replace if needed
        "y": 0.0,  # Replace if needed
        "dx": predicted_action["camera"][1] / CAMERA_SCALER,
        "dy": predicted_action["camera"][0] / CAMERA_SCALER,
        "scaledX": 0.0,  # Replace if needed
        "scaledY": 0.0,  # Replace if needed
        "dwheel": 0.0,
        "buttons": [],  # Populate based on actions
        "newButtons": []
    }

    keyboard = {
        "keys": [key for key, value in predicted_action.items() if (np.array(value) == 1).any() and key != "camera"],
        "newKeys": [],  # Replace if needed
        "chars": ""      # Replace if needed
    }

    return {
        "mouse": mouse,
        "keyboard": keyboard,
        "hotbar": 0,  # Replace if applicable
        "tick": 0,    # Increment tick per frame
        "isGuiOpen": False
    }

def main(model, weights, video_path, output_path, n_frames, n_batches):
    print("Processing video and generating action JSON file...")

    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    required_resolution = ENV_KWARGS["resolution"]
    cap = cv2.VideoCapture(video_path)

    action_data = []
    tick = 0

    for _ in range(n_batches):
        th.cuda.empty_cache()
        print("=== Loading up frames ===")
        frames = []

        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
            # BGR -> RGB
            frames.append(frame[..., ::-1])

        if len(frames) == 0:
            break

        frames = np.stack(frames)
        print("=== Predicting actions ===")
        predicted_actions = agent.predict_actions(frames)

        for i in range(len(frames)):
            predicted_action = {key: predicted_actions[key][0, i] for key in predicted_actions}
            json_action = json_action_from_prediction(predicted_action)
            json_action["tick"] = tick
            action_data.append(json_action)
            tick += 1

    cap.release()

    # Write to JSON file
    with open(output_path, "w") as output_file:
        for action in action_data:
            output_file.write(json.dumps(action) + "\n")

    print(f"Action data saved to {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser("Generate action JSON file from IDM predictions.")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output action JSON file.")
    parser.add_argument("--n-frames", type=int, default=128, help="Number of frames to process at a time.")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches (n-frames) to process.")

    args = parser.parse_args()

    main(args.model, args.weights, args.video_path, args.output_path, args.n_frames, args.n_batches)

