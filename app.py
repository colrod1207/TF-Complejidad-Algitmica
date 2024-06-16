from flask import Flask, render_template, request
from graph_algorithm import create_graph

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/show_graph')
def show_graph():
    plot_url = create_graph()
    return render_template('graph.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
