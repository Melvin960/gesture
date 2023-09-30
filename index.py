# Import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Set up the display in full-screen mode
screen = pygame.display.set_mode((1080,1920), pygame.FULLSCREEN)
pygame.display.set_caption("Hand Gesture Game")
# Get the dimensions of the screen
screen_width, screen_height = pygame.display.get_surface().get_size()
fullscreen=True

# Load available gestures and their corresponding images
gesture_images = {
    "rock": pygame.image.load("rock.jpg"),
    "fist": pygame.image.load("fist.jpg"),
    # ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
    "okay": pygame.image.load("ok.jpg"),
    "peace": pygame.image.load("peace.jpg"),
    "thumbs up": pygame.image.load("thumbs up.jpg"),
   
    "stop": pygame.image.load("stop.jpg"),
    "smile": pygame.image.load("smile.jpg"),
    "live long": pygame.image.load("live long.jpg"),
    
}

# Create a list of available gesture names
available_gestures = list(gesture_images.keys())

# Initialize game variables
score = 0
user_gestures = []
gesture_read_time = 0  # Timestamp when gesture was last read
gesture_cooldown = 2  # Cooldown time before reading a new gesture (in seconds)
feedback_duration = 3  # Increase the feedback duration to 3 seconds
feedback_start_time = 0  # Timestamp when feedback started
game_over = False  # Game over state


# Initialize feedback variables
feedback_color = (0, 0, 0)  # Default feedback color (black)
feedback_text = ""

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Flag to track full-screen mode
fullscreen = False
# Timer variables
countdown_start_time = 0
countdown_duration = 60  # 1 minute
countdown_color = (0, 0, 255)  # Default color (blue)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Check for keyboard events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Check if the Esc key is pressed
                # Exit full-screen mode
                pygame.display.set_mode((800, 600))  # Set the display to a specific size (windowed mode)
                screen = pygame.display.set_mode((800, 600))  # Update the screen variable
                pygame.display.set_caption("Hand Gesture Game")  # Restore the window title
                fullscreen = False
            elif event.key == pygame.K_f:
                if fullscreen:
                    # Exit full-screen mode
                    pygame.display.set_mode((800, 600))  # Set the display to a specific size (windowed mode)
                    screen = pygame.display.set_mode((800, 600))  # Update the screen variable
                    pygame.display.set_caption("Hand Gesture Game")  # Restore the window title
                else:
                    # Enter full-screen mode
                    screen = pygame.display.set_mode((800,600))
                    fullscreen = True


    if not game_over:
        if fullscreen==False:
        # Read each frame from the webcam
            _, frame = cap.read()

        # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
            result = hands.process(frame)

            className = ''

            frame = cv2.resize(frame, (screen_width,screen_height))

            x, y, _ = frame.shape
        else:
            # Read each frame from the webcam
            _, frame = cap.read()

        # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
            result = hands.process(frame)

            className = ''

            frame = cv2.resize(frame, (800,600))

            x, y, _ = frame.shape


        # Post-process the result
        if result.multi_hand_landmarks:
            current_time = time.time()
            elapsed_time = current_time - gesture_read_time
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                # Predict gesture
                prediction = model.predict([landmarks], verbose=0)
                classID = np.argmax(prediction)
                className = classNames[classID]
                gesture_read_time = current_time

        # disply video
        surface = pygame.surfarray.make_surface(frame)
        surface = pygame.transform.rotate(surface, -90)
        screen.blit(surface, (0, 0))

        # Show the prediction on the frame
        pred_font = pygame.font.Font(None, 48)
        pred_font_render = pred_font.render(className, True, (255, 255, 255))
        screen.blit(pred_font_render, (20, 600 - pred_font_render.get_height() - 20))

        # Get the current target gesture
        current_target = available_gestures[score % len(available_gestures)]
        target_image = gesture_images[current_target]
        target_image = pygame.transform.scale(target_image, (300, 300))
        screen.blit(target_image, (0, 0))

      

        # Show feedback animation
        current_time = time.time()
        feedback_elapsed_time = current_time - feedback_start_time
        if feedback_elapsed_time <= feedback_duration:
            feedback_alpha = int(255 - (feedback_elapsed_time / feedback_duration) * 255)
            feedback_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            feedback_surface.fill((feedback_color[0], feedback_color[1], feedback_color[2], feedback_alpha))
            screen.blit(feedback_surface, (0, 0))
            font = pygame.font.Font(None, 48)
            feedback_text_render = font.render(feedback_text, True, (255, 255, 255))
            screen.blit(feedback_text_render, (400 - feedback_text_render.get_width() // 2, 300))

        # Check if the recognized gesture matches the target
        if className == current_target:
            if className not in user_gestures:
                user_gestures.append(className)
                score += 1
                feedback_color = (0, 250, 0)  # Green for correct gesture
                feedback_text = "Correct!"
                feedback_start_time = current_time  # Reset feedback timer for correct gesture
                pygame.draw.rect(screen, feedback_color, (0, 0, screen_width, screen_height))

                # Draw the "Correct!" text
                font = pygame.font.Font(None, 100)
                text_render = font.render(feedback_text, True, (255, 255, 255))
                text_rect = text_render.get_rect(center=(650, 300))
                screen.blit(text_render, text_rect.center)

                pygame.display.flip()  # Update the display

                # Wait for 3 seconds
                time.sleep(3) 
                feedback_color = (0, 0, 0)
                feedback_text = ""

                # Start the countdown timer
                countdown_start_time = current_time
                countdown_color = (0, 0, 255)  # Reset countdown color to blue
           
        else:
            feedback_color = (0, 0, 0)  # Reset feedback color to default (black)
            feedback_text = ""

    # Calculate the remaining time for the countdown timer
        countdown_elapsed_time = current_time - countdown_start_time
        remaining_time = max(0, countdown_duration - countdown_elapsed_time)

        # Change countdown color to red after 50 seconds
        if remaining_time <= 10:
            countdown_color = (255, 0, 0)  # Red color

        # Display the countdown timer
        font = pygame.font.Font(None, 72)
        countdown_text = font.render(f"Time Left: {int(remaining_time)}", True, countdown_color)
        screen.blit(countdown_text, (screen_width // 2 - countdown_text.get_width() // 2, 20))

        # Check if the countdown timer has ended
        if remaining_time <= 0:
            feedback_start_time = 0  # Reset feedback timer
            user_gestures = []  # Clear user gestures
            current_target = random.choice(available_gestures)  # Choose a new random target gesture
            countdown_start_time = current_time  # Restart the countdown timer

    # Game Over screen and Restart button
    else:
        # Display "Game Over" message
        game_over_font = pygame.font.Font(None, 92)
        game_over_text = game_over_font.render("Game Over", True, (255, 0, 0))
        screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, 300))

        # Display "Restart" button
        restart_prompt_font = pygame.font.Font(None, 48)
        restart_prompt_text = restart_prompt_font.render("Press Enter to Restart", True, (0, 255, 0))
        screen.blit(restart_prompt_text, (screen_width // 2 - restart_prompt_text.get_width() // 2, 350))

        # Handle keyboard input to restart the game
        if game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:  # Check if the Enter key is pressed
                # Reset game variables
                score = 0
                user_gestures = []
                game_over = False
                restart_pending = False

    # Draw score on the screen
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (screen_width - score_text.get_width() - 10, 0))

    # Update the Pygame display
    pygame.display.update()

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()