import os
import subprocess
import sys,subprocess
class Opreration_system:

    def __init__(self) -> None:
        self.path = os.path.join(os.path.expanduser('~'), 'Documents')
        pass
    def read_historial(self,name = "\HisotrialBack.txt",data = ""):
        self.create_write_file(data=data)
        powershell_command = [
            'powershell', 
            '-Command', 
            f'Start-Process notepad -ArgumentList "{self.path+name}" -WindowStyle Minimized'
        ]
        process = subprocess.Popen(powershell_command)
    def create_write_file(self,data,name = "\HisotrialBack.txt",mess = True):
        with open(self.path+name,'w') as file:
            file.write(data)
            if mess == True:
                self.mss_info(app_id='Back Propagation',message="Archivo de texto creado en documents",title='Propagation Info',time=3)
       # print(f"{self.path}Adeline.txt")
    def read_file(self):
        with open(self.path+"\HisotrialBack.txt",'r') as lector:
            data = lector.readlines()
        return data
    def search_normal(self,name):
        return os.path.join(os.getcwd(),name)
    def search_doc(self,name):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, name)
    def mss_info(self,title,message,time=5,app_id="Entrenamiento perceptron",image_url = ""):
        try:
            image_url = self.search_doc(name='logo.ico')
        except:
            image_url = self.search_normal(name='logo.ico')
        powershell_command = f"""
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;
            $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastImageAndText02);
            $textNodes = $template.GetElementsByTagName("text");
            $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) > $null;
            $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) > $null;
            """

        powershell_command += f"""
            $imagePath = "{image_url}"
            $imageNodes = $template.GetElementsByTagName("image");
            $imageNodes.Item(0).Attributes.GetNamedItem("src").NodeValue = $imagePath;
            """
            
        powershell_command += f"""
            $toast = [Windows.UI.Notifications.ToastNotification]::new($template);
            $toast.ExpirationTime = [System.DateTimeOffset]::Now.AddSeconds({time});
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("{app_id}");
            $notifier.Show($toast);
            """
        subprocess.run(["powershell", "-Command", powershell_command])


    


