from pipeline.gradio_interface import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()