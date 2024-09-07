import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import os

# 使用 gTTS 进行语音提示
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("afplay output.mp3")  # 使用 macOS 的 afplay 播放音频

speak("Hello, welcome to the Stretch Game!")

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 动作要领与细节
instructions = {
    "Arm Stretch": "Raise your left arm above your head, and stretch it toward the sky. Hold this position.",
    "Leg Stretch": "Stretch your left leg forward, keeping it straight. Hold this position.",
    "Side Bend Stretch": "Raise your left arm and bend your body to the right. Keep your hips steady and hold.",
    "Torso Twist": "Sit up straight, twist your torso to the left, keeping your hips facing forward.",
    "Shoulder Stretch": "Bring your left arm across your chest, use your right arm to pull it closer for a deeper stretch.",
    "Forward Bend": "Bend forward from your hips, try to touch your toes while keeping your legs straight.",
    "Handstand": "Place your hands on the ground, kick your legs up, and hold the handstand position."
}

# 定义多个拉伸动作的标准姿势
poses = [
    {  # 第一个动作：简单手臂伸展
        'name': 'Arm Stretch',
        'landmarks': {
            'left_shoulder': (0.5, 0.5),
            'left_elbow': (0.5, 0.6),
            'left_wrist': (0.5, 0.7),
        },
        'difficulty': 'easy',
        'tolerance': 0.1,  # 宽松的误差范围
        'time_limit': 15,  # 15秒内完成
    },
    {  # 第二个动作：腿部伸展
        'name': 'Leg Stretch',
        'landmarks': {
            'left_hip': (0.5, 0.5),
            'left_knee': (0.5, 0.6),
            'left_ankle': (0.5, 0.7),
        },
        'difficulty': 'medium',
        'tolerance': 0.05,  # 中等误差范围
        'time_limit': 20,  # 20秒内完成
    },
    {  # 第三个动作：侧弯伸展
        'name': 'Side Bend Stretch',
        'landmarks': {
            'left_shoulder': (0.5, 0.4),
            'right_shoulder': (0.5, 0.6),
            'left_hip': (0.4, 0.7),
            'right_hip': (0.6, 0.7),
        },
        'difficulty': 'easy',
        'tolerance': 0.1,
        'time_limit': 15,
    },
    {  # 第四个动作：躯干旋转
        'name': 'Torso Twist',
        'landmarks': {
            'left_shoulder': (0.45, 0.5),
            'right_shoulder': (0.55, 0.5),
            'left_hip': (0.45, 0.7),
            'right_hip': (0.55, 0.7),
        },
        'difficulty': 'easy',
        'tolerance': 0.1,
        'time_limit': 20,
    },
    {  # 第五个动作：肩部伸展
        'name': 'Shoulder Stretch',
        'landmarks': {
            'left_shoulder': (0.4, 0.5),
            'right_shoulder': (0.6, 0.5),
            'left_elbow': (0.4, 0.6),
            'right_elbow': (0.6, 0.6),
        },
        'difficulty': 'easy',
        'tolerance': 0.1,
        'time_limit': 15,
    },
    {  # 第六个动作：前屈伸展
        'name': 'Forward Bend',
        'landmarks': {
            'left_hip': (0.5, 0.5),
            'right_hip': (0.5, 0.5),
            'left_ankle': (0.5, 0.7),
            'right_ankle': (0.5, 0.7),
        },
        'difficulty': 'medium',
        'tolerance': 0.05,
        'time_limit': 20,
    },
    {  # 第七个动作：倒立
        'name': 'Handstand',
        'landmarks': {
            'left_wrist': (0.5, 0.9),
            'left_elbow': (0.5, 0.8),
            'left_shoulder': (0.5, 0.7),
            'left_ankle': (0.5, 0.2),
        },
        'difficulty': 'hard',
        'tolerance': 0.03,
        'time_limit': 30,
    }
]

# 初始化分数
total_score = 0

# 判断用户动作是否接近标准姿势的函数
def check_pose(landmarks, standard_pose, tolerance=0.1):
    def get_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    for key, standard_pos in standard_pose.items():
        joint_pos = (landmarks[getattr(mp_pose.PoseLandmark, key.upper()).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, key.upper()).value].y)
        if get_distance(joint_pos, standard_pos) > tolerance:
            return False
    return True

# 启动摄像头捕捉
cap = cv2.VideoCapture(0)

current_pose = 0  # 当前的拉伸动作索引
start_time = time.time()  # 开始时间

# 提示每个动作的要领
speak(f"Let's start with {poses[current_pose]['name']}! {instructions[poses[current_pose]['name']]}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将图像转换为RGB格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # 获取当前动作的标准姿势和相关参数
    current_stretch = poses[current_pose]
    standard_pose = current_stretch['landmarks']
    action_name = current_stretch['name']
    tolerance = current_stretch['tolerance']
    time_limit = current_stretch['time_limit']
    
    # 如果检测到骨骼
    if results.pose_landmarks:
        # 绘制骨骼关键点
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 获取骨骼的关节点
        landmarks = results.pose_landmarks.landmark
        
        # 检查用户的姿势是否正确
        if check_pose(landmarks, standard_pose, tolerance):
            cv2.putText(frame, f"Pose Correct! Move to next step", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 计算完成时间和分数
            elapsed_time = time.time() - start_time
            score = max(0, (time_limit - elapsed_time) / time_limit * 100)  # 分数基于剩余时间
            total_score += score
            speak(f"Great job! You scored {int(score)} points.")
            
            # 进入下一个动作
            if current_pose < len(poses) - 1:
                current_pose += 1
                start_time = time.time()  # 重置时间
                speak(f"Now let's move to {poses[current_pose]['name']}! {instructions[poses[current_pose]['name']]}")
            else:
                speak("You've completed all the exercises!")
                break
        else:
            cv2.putText(frame, "Adjust your pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # 计算剩余时间
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        speak(f"Time's up! Moving to the next exercise.")
        if current_pose < len(poses) - 1:
            current_pose += 1
            start_time = time.time()  # 重置时间
            speak(f"Now let's move to {poses[current_pose]['name']}! {instructions[poses[current_pose]['name']]}")
        else:
            speak("You've completed all the exercises!")
            break
    
    # 在屏幕上显示当前动作名称和剩余时间
    cv2.putText(frame, f"Current Action: {action_name}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Time Left: {int(remaining_time)}s", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 显示摄像头画面
    cv2.imshow('Stretch Game', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

speak(f"Your total score is {int(total_score)} points.")
