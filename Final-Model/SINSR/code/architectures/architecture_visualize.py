# from architectures.architecture_4 import generator, discriminator
from tensorflow.keras.utils import plot_model # type: ignore
from graphviz import Digraph

def create_generator_diagram():
    dot = Digraph(comment='Multi-Scale Generator Model')

    # Add nodes
    dot.node('A', 'Input\n(192, 256, 3)')
    dot.node('B', 'Conv2D + ReLU\n(64)')
    dot.node('C1', 'Res Block\nScale 1')
    dot.node('C2', 'AveragePool (2x2)\nRes Block\nUpsample (2x)')
    dot.node('C3', 'AveragePool (4x4)\nRes Block\nUpsample (4x)')
    dot.node('D', 'Concatenate\n(192, 256, 192)')
    dot.node('E', 'Conv2D\n(64)')
    dot.node('F1', 'Res Block')
    dot.node('F2', 'Res Block')
    dot.node('F3', 'Res Block')
    dot.node('F4', 'Res Block')
    dot.node('F5', 'Res Block')
    dot.node('G', 'Attention Block\n(64)')
    dot.node('H1', 'UpSampling\n(2x)\nConv2D + ReLU\n(128)')
    dot.node('H2', 'UpSampling\n(2x)\nConv2D + ReLU\n(64)')
    dot.node('I', 'Output\n(768, 1024, 3)')

    # Add edges
    dot.edges(['AB'])
    dot.edge('B', 'C1')
    dot.edge('B', 'C2')
    dot.edge('B', 'C3')
    dot.edge('C1', 'D')
    dot.edge('C2', 'D')
    dot.edge('C3', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F1')
    dot.edge('F1', 'F2')
    dot.edge('F2', 'F3')
    dot.edge('F3', 'F4')
    dot.edge('F4', 'F5')
    dot.edge('F5', 'G')
    dot.edge('G', 'H1')
    dot.edge('H1', 'H2')
    dot.edge('H2', 'I')

    # Save and render the diagram
    dot.render('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Experiments/SINSR/code/architectures/Visualize/multi_scale_generator_diagram', format='png', cleanup=True)
def create_discriminator_diagram():
    dot = Digraph(comment='Enhanced Discriminator Model')

    # Add nodes
    dot.node('A', 'Input\n(768, 1024, 3)')
    dot.node('B', 'Conv2D + ReLU\n(64)')
    dot.node('C1', 'Res Block\nScale 1')
    dot.node('C2', 'AveragePool (2x2)\nRes Block\nResize')
    dot.node('C3', 'AveragePool (4x4)\nRes Block\nResize')
    dot.node('D', 'Concatenate\n(768, 1024, 192)')
    dot.node('E', 'Conv2D + ReLU\n(64)')
    dot.node('F1', 'Conv2D + LeakyReLU\n(128)')
    dot.node('F2', 'Conv2D + LeakyReLU\n(256)')
    dot.node('F3', 'Conv2D + LeakyReLU\n(512)')
    dot.node('F4', 'Conv2D + LeakyReLU\n(512)')
    dot.node('G', 'Final Conv2D\n(1)')

    # Add edges
    dot.edges(['AB'])
    dot.edge('B', 'C1')
    dot.edge('B', 'C2')
    dot.edge('B', 'C3')
    dot.edge('C1', 'D')
    dot.edge('C2', 'D')
    dot.edge('C3', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F1')
    dot.edge('F1', 'F2')
    dot.edge('F2', 'F3')
    dot.edge('F3', 'F4')
    dot.edge('F4', 'G')

    # Save and render the diagram
    dot.render('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Experiments/SINSR/code/architectures/Visualize/enhanced_discriminator_diagram', format='png', cleanup=True)

from graphviz import Digraph

def create_fancy_generator_diagram():
    dot = Digraph(comment='Fancy Multi-Scale Generator Model')

    # Add nodes with fancy styles
    dot.node('A', 'Input\n(192, 256, 3)', shape='box', style='filled', color='lightgrey')
    dot.node('B', 'Conv2D + ReLU\n(64)', shape='box', style='filled', color='lightblue')
    dot.node('C1', 'Res Block\nScale 1', shape='box', style='filled', color='lightgreen')
    dot.node('C2', 'AveragePool (2x2)\nRes Block\nUpsample (2x)', shape='box', style='filled', color='lightgreen')
    dot.node('C3', 'AveragePool (4x4)\nRes Block\nUpsample (4x)', shape='box', style='filled', color='lightgreen')
    dot.node('D', 'Concatenate\n(192, 256, 192)', shape='box', style='filled', color='orange')
    dot.node('E', 'Conv2D\n(64)', shape='box', style='filled', color='lightblue')
    dot.node('F1', 'Res Block', shape='box', style='filled', color='lightgreen')
    dot.node('F2', 'Res Block', shape='box', style='filled', color='lightgreen')
    dot.node('F3', 'Res Block', shape='box', style='filled', color='lightgreen')
    dot.node('F4', 'Res Block', shape='box', style='filled', color='lightgreen')
    dot.node('F5', 'Res Block', shape='box', style='filled', color='lightgreen')
    dot.node('G', 'Attention Block\n(64)', shape='box', style='filled', color='yellow')
    dot.node('H1', 'UpSampling\n(2x)\nConv2D + ReLU\n(128)', shape='box', style='filled', color='lightblue')
    dot.node('H2', 'UpSampling\n(2x)\nConv2D + ReLU\n(64)', shape='box', style='filled', color='lightblue')
    dot.node('I', 'Output\n(768, 1024, 3)', shape='box', style='filled', color='lightgrey')

    # Add edges with fancy styles
    dot.edge('A', 'B', color='black')
    dot.edge('B', 'C1', color='black')
    dot.edge('B', 'C2', color='black')
    dot.edge('B', 'C3', color='black')
    dot.edge('C1', 'D', color='black')
    dot.edge('C2', 'D', color='black')
    dot.edge('C3', 'D', color='black')
    dot.edge('D', 'E', color='black')
    dot.edge('E', 'F1', color='black')
    dot.edge('F1', 'F2', color='black')
    dot.edge('F2', 'F3', color='black')
    dot.edge('F3', 'F4', color='black')
    dot.edge('F4', 'F5', color='black')
    dot.edge('F5', 'G', color='black')
    dot.edge('G', 'H1', color='black')
    dot.edge('H1', 'H2', color='black')
    dot.edge('H2', 'I', color='black')

    # Save and render the diagram
    dot.render('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Experiments/SINSR/code/architectures/Visualize/fancy_multi_scale_generator_diagram', format='png', cleanup=True)

def create_fancy_discriminator_diagram():
    dot = Digraph(comment='Fancy Enhanced Discriminator Model')

    # Add nodes with fancy styles
    dot.node('A', 'Input\n(768, 1024, 3)', shape='box', style='filled', color='lightgrey')
    dot.node('B', 'Conv2D + ReLU\n(64)', shape='box', style='filled', color='lightblue')
    dot.node('C1', 'Res Block\nScale 1', shape='box', style='filled', color='lightgreen')
    dot.node('C2', 'AveragePool (2x2)\nRes Block\nResize', shape='box', style='filled', color='lightgreen')
    dot.node('C3', 'AveragePool (4x4)\nRes Block\nResize', shape='box', style='filled', color='lightgreen')
    dot.node('D', 'Concatenate\n(768, 1024, 192)', shape='box', style='filled', color='orange')
    dot.node('E', 'Conv2D + ReLU\n(64)', shape='box', style='filled', color='lightblue')
    dot.node('F1', 'Conv2D + LeakyReLU\n(128)', shape='box', style='filled', color='red')
    dot.node('F2', 'Conv2D + LeakyReLU\n(256)', shape='box', style='filled', color='red')
    dot.node('F3', 'Conv2D + LeakyReLU\n(512)', shape='box', style='filled', color='red')
    dot.node('F4', 'Conv2D + LeakyReLU\n(512)', shape='box', style='filled', color='red')
    dot.node('G', 'Final Conv2D\n(1)', shape='box', style='filled', color='purple')

    # Add edges with fancy styles
    dot.edge('A', 'B', color='black')
    dot.edge('B', 'C1', color='black')
    dot.edge('B', 'C2', color='black')
    dot.edge('B', 'C3', color='black')
    dot.edge('C1', 'D', color='black')
    dot.edge('C2', 'D', color='black')
    dot.edge('C3', 'D', color='black')
    dot.edge('D', 'E', color='black')
    dot.edge('E', 'F1', color='black')
    dot.edge('F1', 'F2', color='black')
    dot.edge('F2', 'F3', color='black')
    dot.edge('F3', 'F4', color='black')
    dot.edge('F4', 'G', color='black')
    
    # Save and render the diagram
    dot.render('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Experiments/SINSR/code/architectures/Visualize/fancy_enhanced_discriminator_diagram', format='png', cleanup=True)

create_fancy_discriminator_diagram()
print('hy')
create_fancy_generator_diagram()


# Build and visualize the generator model
# super_resolution_generator = generator()
# plot_model(super_resolution_generator, to_file='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Experiments/SINSR/code/architectures/Visualize/multi_scale_generator.png', show_shapes=True, show_layer_names=True)

# hybrid_discriminator = discriminator(input_shape=(768, 1024, 3))
# plot_model(hybrid_discriminator, to_file='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Experiments/SINSR/code/architectures/Visualize/enhanced_discriminator.png', show_shapes=True, show_layer_names=True)

# create_discriminator_diagram()
# create_generator_diagram()