import threading
from datetime import datetime

import urwid


class SubmitEdit(urwid.Edit):
    signals = ["submit"]

    def keypress(self, size, key):
        if key == "enter":
            self._emit("submit")
            return None
        else:
            return super().keypress(size, key)

    def insert_text(self, text):
        self.edit_text += text
        self.edit_pos = len(self.edit_text)


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")


class ChatApp(urwid.WidgetWrap):
    def __init__(self, model, tokenizer, inference_fn, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.inference_fn = inference_fn
        self.kwargs = kwargs

        # Chat history and input field
        self.chat_history = urwid.SimpleListWalker([])
        self.chat_box = urwid.ListBox(self.chat_history)
        self.input_field = SubmitEdit("> ", multiline=False)

        # Add initial system message with updated instructions
        disclaimer = (
            "Hello, I am Kurtis, a transformer-based AI/ML model for mental health support.\n"
            "This tool does not substitute professional mental health services.\n"
            "Consult a licensed professional for urgent concerns.\n\n"
            "Instructions:\n"
            "- Type your message below.\n"
            "- You can paste multi-line text into the input field.\n"
            "- Press **Enter** to submit your message.\n"
            "- Press **Esc** to exit the application."
        )
        self.add_message(f"[{get_timestamp()}] System:\n{disclaimer}", "system")

        # Create layout with chat history and input, set focus to footer
        self.main_view = urwid.Frame(
            urwid.AttrMap(self.chat_box, "chat"),
            footer=urwid.AttrMap(self.input_field, "input"),
            focus_part="footer",  # Ensure the input field has focus
        )

        super().__init__(self.main_view)  # Wrap the main_view widget

        # Connect the 'submit' signal from the input field to the handle_input method
        urwid.connect_signal(self.input_field, "submit", self.handle_input)

    def add_message(self, message, style):
        """Add a new message to the chat history and refresh the UI."""
        message_widget = urwid.Text(message)
        self.chat_history.append(urwid.AttrMap(message_widget, style))
        # Auto-scroll to the latest message
        self.chat_box.set_focus(len(self.chat_history) - 1)

    def handle_input(self, widget=None):
        """Handle user input and generate a response."""
        input_text = self.input_field.get_edit_text()
        if input_text.strip():
            timestamp = get_timestamp()
            user_message = f"[{timestamp}] You: {input_text}"
            self.add_message(user_message, "user")

            # Add loading cursor
            loading_message = f"[{get_timestamp()}] Kurtis: ... (thinking)"
            loading_widget = urwid.Text(loading_message)
            loading_map = urwid.AttrMap(loading_widget, "bot")
            self.chat_history.append(loading_map)
            self.chat_box.set_focus(len(self.chat_history) - 1)

            # Run inference in a separate thread
            threading.Thread(
                target=self.generate_response,
                args=(input_text, len(self.chat_history) - 1),
                daemon=True,
            ).start()

            # Clear the input field
            self.input_field.set_edit_text("")
            urwid.emit_signal(self.input_field, "change")

    def generate_response(self, input_text, loading_index):
        """Generate response in a separate thread."""
        try:
            response = self.inference_fn(
                self.model, self.tokenizer, input_text, **self.kwargs
            )
            ai_message = f"[{get_timestamp()}] Kurtis: {response}"
            self.chat_history[loading_index] = urwid.AttrMap(
                urwid.Text(ai_message), "bot"
            )
        except Exception as e:
            error_message = f"[{get_timestamp()}] Kurtis: (Error: {str(e)})"
            self.chat_history[loading_index] = urwid.AttrMap(
                urwid.Text(error_message), "bot"
            )

        # Refresh the UI after updating the message
        self.main_view.body._invalidate()

    def unhandled_input(self, key):
        """Handle global key inputs."""
        if key == "esc":
            raise urwid.ExitMainLoop()

    def run(self):
        """Run the chat application."""
        palette = [
            ("input", "light cyan", "black"),
            ("user", "light green", "black"),
            ("system", "light red", "black"),
            ("bot", "light blue", "black"),
            ("chat", "white", "black"),
        ]
        screen = urwid.raw_display.Screen()
        screen.use_alternate_screen = False  # Disable alternate screen buffer
        loop = urwid.MainLoop(
            self,
            palette,
            screen=screen,
            unhandled_input=self.unhandled_input,
            handle_mouse=False,
        )
        loop.run()


# Start the chat application
def start_chat_wrapper(model, tokenizer, inference_fn, **kwargs):
    app = ChatApp(model, tokenizer, inference_fn, **kwargs)
    app.run()
