import cv2
import sys

"""
Click two points on the image; prints pixel distance.
Controls:
  - Left click: add point
  - '+' / '-': zoom in/out
  - '1': 100% (actual size)
  - 'f': fit to screen
  - Drag with right mouse: pan
Usage:
  python measure_px_length.py path/to/image.png
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python measure_px_length.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    orig = cv2.imread(img_path)
    if orig is None:
        print(f"Failed to load image: {img_path}")
        sys.exit(1)

    # Display state
    zoom = 1.0
    offset = [0, 0]  # pan offset in display pixels
    dragging = False
    drag_start = (0, 0)
    pts_orig = []  # points in original image coordinates

    # Compute fit-to-screen zoom for initial view
    screen_w = 1280
    screen_h = 720
    zh = screen_h / orig.shape[0]
    zw = screen_w / orig.shape[1]
    fit_zoom = min(zh, zw)
    zoom = fit_zoom

    def render():
        disp = cv2.resize(orig, (int(orig.shape[1]*zoom), int(orig.shape[0]*zoom)), interpolation=cv2.INTER_LINEAR)
        view = disp.copy()
        # Draw points/lines
        for i, (xo, yo) in enumerate(pts_orig):
            xd = int(xo * zoom)
            yd = int(yo * zoom)
            cv2.circle(view, (xd, yd), 5, (0, 0, 255), -1)
            cv2.putText(view, f"({xo},{yo})", (xd+8, yd-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        if len(pts_orig) >= 2:
            x1, y1 = pts_orig[0]
            x2, y2 = pts_orig[1]
            d = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            cv2.line(view, (int(x1*zoom), int(y1*zoom)), (int(x2*zoom), int(y2*zoom)), (0,255,0), 2)
            tx = min(int(x1*zoom), int(x2*zoom)) + 10
            ty = min(int(y1*zoom), int(y2*zoom)) + 25
            cv2.putText(view, f"{d:.2f} px", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('measure', view)

    def to_orig_coords(xd, yd):
        # Map display coords to original image coords
        xo = int(round(xd / zoom))
        yo = int(round(yd / zoom))
        xo = max(0, min(orig.shape[1]-1, xo))
        yo = max(0, min(orig.shape[0]-1, yo))
        return xo, yo

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, drag_start, offset, pts_orig
        if event == cv2.EVENT_LBUTTONDOWN:
            xo, yo = to_orig_coords(x, y)
            pts_orig.append((xo, yo))
            if len(pts_orig) == 2:
                x1, y1 = pts_orig[0]
                x2, y2 = pts_orig[1]
                d = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                print(f"Pixel distance: {d:.2f} px")
                print("Press any key to exit or continue measuring.")
            render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            offset[0] += dx
            offset[1] += dy
            drag_start = (x, y)
            render()
        elif event == cv2.EVENT_RBUTTONUP:
            dragging = False

    cv2.namedWindow('measure', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('measure', on_mouse)
    render()
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord('+') or k == ord('='):
            zoom = min(8.0, zoom * 1.25)
            render()
        elif k == ord('-') or k == ord('_'):
            zoom = max(0.1, zoom / 1.25)
            render()
        elif k == ord('1'):
            zoom = 1.0
            render()
        elif k == ord('f'):
            zoom = fit_zoom
            render()
        elif k == ord('c'):
            pts_orig = []
            render()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
