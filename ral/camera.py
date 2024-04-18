import cv2

cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Read a new frame
    ret, board_img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("board", board_img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the capture and destroy all windows when done
cap.release()
cv2.destroyAllWindows()