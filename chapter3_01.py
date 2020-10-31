import tensorflow as tf

# 그래프의 노드를 정의하고 출력함
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)    # 암시적으로 tf.float32 타입으로 선언됨.
print(node1, node2)

# 그래프를 실행함
print(node1.numpy(), node2.numpy())

# 두개의 노드의 값을 더하는 연산을 수행하는 node3을 정의합니다.
node3 = tf.add(node1, node2)
print('node3 :', node3)
print(f'node3.numpy() : {node3.numpy()}')

tensor1 = tf.constant([1, 2, 3, 4, 5, 6, 7])
tensor2 = tf.constant(-1.0, shape=[2, 3])
print(tensor1)
print(tensor2)