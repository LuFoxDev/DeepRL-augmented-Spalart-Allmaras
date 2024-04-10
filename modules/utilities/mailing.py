import smtplib
import ssl
from email.message import EmailMessage
import os
import mimetypes
import json

def send_attachment_via_email(attachment_path, subject, body):
    try:
        path_to_email_credentials = "/home/lukas/Documents/email_service_credentials.json"
        
        with open(path_to_email_credentials) as json_file:
            email_credentials = json.load(json_file)

        email_sender = email_credentials["email_sender"]
        app_password = email_credentials["app_password"]
        email_receiver = email_credentials["email_receiver"]

        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = email_receiver
        em['Subject'] = subject
        em.set_content(body)

        mime_type, _ = mimetypes.guess_type(attachment_path)
        mime_type, mime_subtype = mime_type.split('/', 1)
        with open(attachment_path, 'rb') as ap:
            em.add_attachment(ap.read(), maintype=mime_type, subtype=mime_subtype,
                filename=os.path.basename(attachment_path))

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(email_sender, app_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())

    except Exception:
        print("sending mail failed!")

if __name__ == "__main__":

    attachment_path = "/home/lukas/Code/metmo/exports/RL_V9_10by10_0.02m_linear_discount0.4_clip0.9_run3/post-processing/rmse_by_time.png"
    subject = "Test E-Mail"
    body = "This is a test email that contains an image"

    send_attachment_via_email(attachment_path, subject, body)
