from manim import *
import random

class NeuralNetwork(Scene):
    def construct(self):
        # Define network layers and nodes
        fill_op = 1
        inactive = WHITE
        active = GREEN
        wrong = RED
        input_nodes = VGroup(
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op)
        ).arrange(direction=UP, buff=1).shift(LEFT*3)

        hidden_nodes = VGroup(
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op)
        ).arrange(direction=UP, buff=1.1)
        
        weights = VGroup(
            DecimalNumber(0.48).scale(0.7).next_to(hidden_nodes[0], UP),
            DecimalNumber(0.24).scale(0.7).next_to(hidden_nodes[1], UP),
            DecimalNumber(0.91).scale(0.7).next_to(hidden_nodes[2], UP)
        )

        output_nodes = VGroup(
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op),
            Circle(radius= 0.4, color=inactive, fill_opacity=fill_op)
        ).arrange(direction=UP, buff=2).shift(RIGHT * 3)
        
        labels = VGroup(
            Text("Wrong", color=RED).next_to(output_nodes[1], RIGHT),
            Text("Right", color=GREEN).next_to(output_nodes[0], RIGHT),
            Text("Input", color=YELLOW).next_to(input_nodes[2], LEFT * 2),
        )
        # Connect network layers
        connections = VGroup()
        for i in input_nodes:     #lines 0-14
            for h in hidden_nodes:
                connections.add(Line(i,h,color=inactive, stroke_width=5))
        for i in hidden_nodes:    #lines 15-20
            for h in output_nodes:
                connections.add(Line(i,h,color=inactive, stroke_width=5))
        # Add network elements to scene
        self.play(Create(input_nodes), Create(output_nodes), Create(hidden_nodes))
        self.play(Create(connections))
        self.play(Create(weights), Create(labels), runtime=0.3)

        #Activate input layer
        self.play(input_nodes[0].animate.set_color(YELLOW),
                  input_nodes[1].animate.set_color(YELLOW),
                  input_nodes[3].animate.set_color(YELLOW))

        #Activate layer 1
        self.play(connections[0].animate.set_color(active),
                  connections[1].animate.set_color(active),               #input_node 0 -> hidden_node 1
                  connections[2].animate.set_color(active),
                  connections[3].animate.set_color(active),
                  connections[4].animate.set_color(active),               #input_node 1 -> hidden_node 1
                  connections[5].animate.set_color(active),
                  connections[9].animate.set_color(active),
                  connections[10].animate.set_color(active),
                  connections[11].animate.set_color(active), run_time=0.5) #input_node 2 -> hidden_node 0

        #Activate hidden nodes
        self.play(hidden_nodes[0].animate.set_color(active),
                  hidden_nodes[1].animate.set_color(active),
                  hidden_nodes[2].animate.set_color(active),
                  run_time=1)

        #Activate layer 2
        self.play(connections[15].animate.set_color(active),               #hidden_node 1 -> output_node 1
                  connections[16].animate.set_color(active),
                  connections[17].animate.set_color(active),
                  connections[18].animate.set_color(active),
                  run_time=0.5) #input_node 0 -> output_node 1
        
        #Activate output
        self.play(output_nodes[1].animate.set_color(wrong))
        self.wait(0.5)
        self.play(ChangeDecimalToValue(weights[1], 0.12),
                  ChangeDecimalToValue(weights[0], 0.35))
        self.wait(1)
        #Reset network
        self.play(input_nodes[0].animate.set_color(inactive),
                  input_nodes[1].animate.set_color(inactive),
                  input_nodes[3].animate.set_color(inactive),
                  hidden_nodes[0].animate.set_color(inactive),
                  hidden_nodes[1].animate.set_color(inactive),
                  hidden_nodes[2].animate.set_color(inactive),      
                  connections[0].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[1].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[2].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[3].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[4].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[5].animate.set_color(inactive),       #input_node 1 -> hidden_node 1
                  connections[9].animate.set_color(inactive),       #input_node 2 -> hidden_node 0
                  connections[10].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[11].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[15].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[16].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[17].animate.set_color(inactive),       #input_node 0 -> hidden_node 1
                  connections[18].animate.set_color(inactive),      #hidden_node 1 -> output_node 1
                  connections[19].animate.set_color(inactive),      #input_node 0 -> output_node 1
                  connections[20].animate.set_color(inactive),      #input_node 0 -> output_node 1
                  output_nodes[1].animate.set_color(inactive),
                  run_time=0.5)
            
        self.wait(1)
