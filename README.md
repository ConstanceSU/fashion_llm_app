## What is the utility of the prototype?

The Virtual Closet and Outfit Simulator is an innovative web application designed to help users
manage their wardrobe and receive personalized outfit recommendations. By integrating image
recognition, color detection, and language model capabilities, this app streamlines the process of
organizing clothing items and suggests stylish combinations based on users’ upcoming events or
personal preferences.

## What are the main design decisions chosen?
- Image Recognition & Tagging: Utilize deep learning (MobileNetV2) to identify and classify
clothing items uploaded by users.
- Color Detection: Enhance basic classification by detecting dominant colors, allowing for more
detailed fashion attributes.
- Virtual Wardrobe Management: Enable users to build a digital wardrobe where they can add,
rename, and remove clothing items.
- Outfit Recommendation: Leverage a language model (via Cohere API) to generate tailored outfit
suggestions based on user-input style prompts and the contents of their wardrobe.

## Main difficulties found?
• Integration of the webcolors library presented significant challenges. Despite
multiple trials, the library did not function as expected in my Python environment.
As a result, I developed a custom solution by manually constructing a mapping of
CSS3 color names to their corresponding RGB values. This approach ensured a
more reliable and consistent detection of dominant colors in clothing images.

• UX design difficulties
Although the incorporation of the LLM component (via the Cohere API) was
relatively straightforward, a considerable amount of time and effort was dedicated
to refining the user interface. The UX design posed substantial challenges, as it
required iterative adjustments to achieve an intuitive, engaging, and aesthetically
pleasing experience. Balancing functionality with ease of use was critical, and the
design process involved extensive testing and refinement to meet the desired user
experience standards.

• Advanced Clothing Recognition Integration:
In an effort to enhance the accuracy of clothing item identification, I explored the
integration of Fashion-CLIP, a model specifically tailored for fashion-related image
recognition. However, I encountered resource and time constraints, as fine-tuning
Fashion-CLIP requires access to a very large dataset and significant computational
resources. Although I was unable to fully implement this approach within the scope
of the current project, the potential of Fashion-CLIP to provide more precise and
detailed recognition remains an intriguing prospect for future research.
