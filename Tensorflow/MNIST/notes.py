#options for running python program in VS Code:
    #1) [easy] Command Pallate (Ctrl+Shift+P) => Run Python File in Terminal
    #[Others] 

import tensorflow as tf


print("Starting program..")
sess = tf.Session()

#the computation graph is composed of these elements:
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.constant(32, dtype=tf.float32)

#What kind of computation do we want to perform?
computation = a*x + y*y


#"feed in" values for the unknowns (inputs, ..) 
#and run it in a Tensorflow Session
first_feed = { x:2, y:4 }
print(sess.run(computation, first_feed))

#same thing as above, but with different
#"feed values"
second_feed = { x: [1,2,3], y: [2,3,4]}
print(sess.run(computation, second_feed))

sess.close()


print("second try..")
#lets try that again.. from starting point

#1. Create a session on which the computation
#will be performed.
sess = tf.Session()

#2. Define the "bare" elements of the computation
W = tf.Variable(        [2], dtype=tf.float32)

b = tf.Variable(        [3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
feed_input_for_x = {x:  [4,5,1,8]}

#3. Define the computation itself.
someShittyLinearFunction = W*x + b

#the variables need their values filled in./
#just like how we "feed" inputs to placeholders,
#we need to "initialize" variables..

sess.run(tf.global_variables_initializer())

#now that the variables have their values,
#we can run the computation we want..
print(sess.run(someShittyLinearFunction, feed_input_for_x))
sess.close()










#Nice! Lets try that again, but this time the computation
#that I want to perform is the Logistic Regression
#algorithms