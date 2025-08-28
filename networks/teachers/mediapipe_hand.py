import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def hand_pose_img(image):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        image_height, image_width, _ = image.shape

    return [results, image.shape]


def hand_pose_cam(cap):
    # For webcam input:
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break


def hand_pose_video(image):
    # For webcam input:
    posed_image = image.copy()
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        # To improve performance, optionally mark the posed_image as not
        # writeable to pass by reference.
        posed_image.flags.writeable = False
        posed_image = cv2.cvtColor(posed_image, cv2.COLOR_BGR2RGB)
        results = hands.process(posed_image)

        # Draw the hand annotations on the posed_image.
        posed_image.flags.writeable = True
        posed_image = cv2.cvtColor(posed_image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    posed_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        return posed_image


def pose_demo_img(image_file_list):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for idx, file in enumerate(image_file_list):
            print(f"processing {file}")
            image = cv2.imread(file)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            print(results)
            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print("# Draw the hand annotations on the image.")
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            return image
            # Flip the image horizontally for a selfie-view display.
            # cv2.imwrite("/tmp/pose_img", image)
            # cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
            # print("saved")
            # cv2.waitKey(0)

