import cv2
from common_utils import utilities

class PostProcess:

    def __init__(self,

                 device,

                 config_file,

                 ) -> None:

        try:
            self.device = device
            self.config = config_file
            self.utils = utilities()
            self.classes_dict = self.utils.getConfigParam('threshold_dict',self.config['camera_list'][0])
            self.class_names = self.utils.read_classes_file(self.utils.getConfigParam('classes',self.config['camera_list'][0]))

        except Exception as exp:
            print(str(exp))

    def check_Helmet_within_Person_Coords(self, PPE_rects, person_rect):
        PPE_status = False
        PPE_tracked = None
        try:
            for i, rect in enumerate(PPE_rects):
                startX, startY, endX, endY = rect
                pt = (round((startX + endX) / 2, 2), round((startY + endY) / 2, 2))
                logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3] / 2 + person_rect[1] / 2
                if logic == True:
                    PPE_status = True
                    PPE_tracked = rect
                    break
        except:
            print("error while mapping the Helmet")

        return PPE_status, PPE_tracked



    def check_PPE_within_Person_Coords(self,PPE_rects,person_rect):
        PPE_status = False
        PPE_tracked = None
        try:
            for i,rect in enumerate(PPE_rects):
                startX, startY, endX, endY = rect
                pt = (round((startX+endX)/2,2),round((startY+endY)/2,2))
                logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3]
                if logic==True:
                    PPE_status = True
                    PPE_tracked=rect
                    break

        except:
            print("error while mapping the PPE")
        return PPE_status,PPE_tracked


    def check_shoe_within_Person_Coords(self, PPE_rects, person_rect):
        PPE_status = False
        PPE_tracked = None
        try:
            for i, rect in enumerate(PPE_rects):
                startX, startY, endX, endY = rect
                pt = (round((startX + endX) / 2, 2), round((startY) / 2, 2))
                logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3]
                if logic == True:
                    PPE_status = True
                    PPE_tracked = rect
                    break
        except:
            print("error while mapping the Shoe")
        return PPE_status, PPE_tracked

    def check_PPE_within_person(self, frame, coords):
        if any(coords["person"]):
                for person_rect in coords["person"]:
                    PPE_list = []
                    keysfound = [key for key in coords.keys() if key != "person"]
                    if 'helmet' in keysfound:
                        with_helmet_status, helmet_coords = self.check_Helmet_within_Person_Coords(coords['helmet'], person_rect)
                        if with_helmet_status == True:
                            PPE_list.append('helmet')
                    if 'shoe' in keysfound:
                        with_shoe_status, shoe_coords = self.check_shoe_within_Person_Coords(coords['shoe'], person_rect)
                        if with_shoe_status == True:
                            PPE_list.append('shoe')
                    if 'vest' in keysfound:
                        with_vest_status, vest_coords = self.check_PPE_within_Person_Coords(coords['vest'], person_rect)
                        if with_vest_status == True:
                            PPE_list.append('vest')

                    if len(PPE_list) > 0:
                        (startX, startY, endX, endY) = person_rect
                        pt1 = int(startX), int(startY)
                        pt2 = int(endX), int(endY)
                        text_x = int(startX) + 4
                        text_y = int(startY) - 10
                        coords_keys = [key for key in coords.keys() if key != "person"]
                        ppe_list = PPE_list
                        clean_coords_keys = [key.strip() for key in coords_keys]
                        clean_ppe_list = [item.strip() for item in ppe_list]
                        sorted_clean_coords_keys = sorted(clean_coords_keys)
                        sorted_clean_ppe_list = sorted(clean_ppe_list)
                        if sorted_clean_coords_keys == sorted_clean_ppe_list:
                            color = (0,255,0)
                            cv2.rectangle(frame, pt1, pt2, color, 2)
                            cv2.putText(frame, 'PPE Compliance', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            color = (0,0,255)
                            cv2.rectangle(frame, pt1, pt2, color, 2)
                            cv2.putText(frame, 'PPE Non-Compliance', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame