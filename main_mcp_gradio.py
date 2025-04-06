# main.py
import asyncio
import gradio as gr
import os
from dotenv import load_dotenv

from mcpfunction import run_mcp

if __name__ == "__main__":
    # message = "Look at my local files and find what my favorite city is for fall."
    # final_output = asyncio.run(run_mcp(message))
    # print(f"\nFinal Output Returned to Main:\n{final_output}")
    try:
        with gr.Blocks() as iface:
            with gr.Row():
                gr.HTML("<h3 style='margin-left: -11px;'>Company Intranet</h3>")
            with gr.Row():
                output_textbox = gr.Textbox(label="Chatbot:")
            with gr.Row():
                input_question = gr.Textbox(label="Your Question:")
            with gr.Row():
                gr.ClearButton(components=input_question)
                btn_submit = gr.Button("Submit")
                btn_submit.variant = "primary"
                btn_submit.click(fn=run_mcp, inputs=input_question, outputs=output_textbox)
    
        iface.launch()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
