import cv2
import numpy as np

# Function to draw a single footprint
def draw_right_footprint(img, center, size, angle, color=(0, 0, 0), thickness=2):
    foot_length = size
    foot_width = size // 2
    toe_radius = size // 6
    heel_radius = size // 3

    # Center points for the heel and the front part of the foot (metatarsal heads)
    heel_center = (
        int(center[0] - foot_length * np.cos(angle)),
        int(center[1] + foot_length * np.sin(angle))
    )
    
    # Positions for individual toes
    toe_centers = [
        (
            int(center[0] + foot_length * np.cos(angle) + toe_radius * i * np.cos(angle + np.pi / 4)),
            int(center[1] - foot_length * np.sin(angle) - toe_radius * i * np.sin(angle + np.pi / 4))
        )
        for i in range(-2, 3)
    ]
    
    # Draw the heel
    cv2.ellipse(img, heel_center, (heel_radius, heel_radius // 2), np.degrees(angle), 0, 360, color, thickness)
    
    # Draw the toes
    for toe_center in toe_centers:
        cv2.circle(img, toe_center, toe_radius, color, thickness)
    
    # Draw the main foot shape connecting heel to toes
    points = [heel_center] + toe_centers
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)

# Function to draw a right footprint
def draw_left_footprint(img, center, size, angle, color=(0, 0, 0), thickness=2):
    foot_length = size
    foot_width = size // 2
    toe_radius = size // 6
    heel_radius = size // 3

    # Center points for the heel and the front part of the foot (metatarsal heads)
    heel_center = (
        int(center[0] - foot_length * np.cos(angle)),
        int(center[1] + foot_length * np.sin(angle))
    )
    
    # Positions for individual toes
    toe_centers = [
        (
            int(center[0] + foot_length * np.cos(angle) + toe_radius * i * np.cos(angle - np.pi / 4)),
            int(center[1] - foot_length * np.sin(angle) - toe_radius * i * np.sin(angle - np.pi / 4))
        )
        for i in range(-2, 3)
    ]
    
    # Draw the heel
    cv2.ellipse(img, heel_center, (heel_radius, heel_radius // 2), np.degrees(angle), 0, 360, color, thickness)
    
    # Draw the toes
    for toe_center in toe_centers:
        cv2.circle(img, toe_center, toe_radius, color, thickness)
    
    # Draw the main foot shape connecting heel to toes
    points = [heel_center] + toe_centers
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)



img = np.ones((700, 700, 3), dtype=np.uint8) * 255

footfallsLeft = [[ 0.4224, -0.5520, -0.1362], [ 1.5806, -0.7751,  0.1331], [ 2.7451, -0.8331,  0.0467], [ 3.8020, -0.9480, -0.1968]]
footfallsRight = [[ 0.3359, -0.7139,  0.0480], [ 0.9321, -0.7475,  0.1890], [ 2.1968, -0.8167,  0.3034], [ 3.1120, -0.8721,  0.1730], [ 3.8717, -0.9334, -0.1534]]



maxX = -999
minX = 999
maxY = -999
minY = 999
for foot in footfallsLeft:
    if(foot[0] > maxX):
        maxX = foot[0]
    elif(foot[0] < minX):
        minX = foot[0]

    if(foot[2] > maxY):
        maxY = foot[2]
    elif(foot[2] < minY):
        minY = foot[2]
for foot in footfallsRight:
    if(foot[0] > maxX):
        maxX = foot[0]
    elif(foot[0] < minX):
        minX = foot[0]

    if(foot[2] > maxY):
        maxY = foot[2]
    elif(foot[2] < minY):
        minY = foot[2]


varX = maxX - minX
varY = maxY - minY

maxVar = varX if varX > varY else varY

for footfall in footfallsLeft:
    footfall[0] -= minX
    footfall[2] -= minY

for footfall in footfallsRight:
    footfall[0] -= minX
    footfall[2] -= minY



scale = 600/varX

for footfall in footfallsLeft:
    draw_left_footprint(img, [int(footfall[0] * scale) + 50, int(footfall[2] * scale) + 50], 15, 0)

for footfall in footfallsRight:
    draw_right_footprint(img, [int(footfall[0] * scale) + 50, int(footfall[2] * scale) + 50], 15, 0)

cv2.imshow('Footsteps', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
