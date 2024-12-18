from flask import Flask, send_from_directory

app = Flask(__name__)

# Route to serve image1
@app.route('/image1')
def serve_image1():
    return send_from_directory('static/images', 'Image1.png')

# Route to serve image2
@app.route('/image2')
def serve_image2():
    return send_from_directory('static/images', 'Image2.png')

# Root route that displays both images
@app.route('/')
def index():
    return '''
    <h1>Image Server</h1>
    <img src="/image1" alt="Image 1">
    <img src="/image2" alt="Image 2">
    '''

if __name__ == '__main__':
    app.run(debug=True)