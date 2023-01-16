from flask import Flask, jsonify, request, render_template
import final

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def testfn():
   message = {'btc_price':final.bitcoin(), 'eth_price':final.eth()}
   
   return jsonify(message)

if __name__ == "__main__":
   app.run(debug=True)
